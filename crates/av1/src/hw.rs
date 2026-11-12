//! Header parsing for stateless hardware decoders (VA-API): the OBU walk, the uncompressed frame
//! header with its cross-frame state (7.20, 7.21) and the tile data of each frame, without decoding
//! any picture. It is the software decoder's own parser, so both decide alike.
//!
//! A hardware front end feeds [`HeaderParser::parse`] the same chunks the software [`crate::Decoder`]
//! gets and receives the frames in decoding order: each one to decode ([`Unit::Frame`]) or to show
//! again ([`Unit::Show`]). The reference slots a frame refreshes are in its header.

use crate::bits::{BitReader, leb128};
use crate::header::{HeaderState, RefInfo, default_gm_params};
use crate::inter::setup_shear;
use crate::spec_tables::{ALTREF_FRAME, KEY_FRAME, LAST_FRAME, NUM_REF_FRAMES, TRANSLATION};
use crate::{Error, Result, SequenceHeader};

pub use crate::header::{CdefParams, FilmGrainParams, FrameHeader, LoopFilterDeltas, LoopFilterParams, LrParams, QuantParams, Segmentation, TileInfo};

const OBU_SEQUENCE_HEADER: u8 = 1;
const OBU_TEMPORAL_DELIMITER: u8 = 2;
const OBU_FRAME_HEADER: u8 = 3;
const OBU_TILE_GROUP: u8 = 4;
const OBU_FRAME: u8 = 6;
const OBU_REDUNDANT_FRAME_HEADER: u8 = 7;

/// Most frames one chunk may hold (a temporal unit holds a few).
const MAX_UNITS_PER_CHUNK: usize = 256;

/// The coded data of one tile.
#[derive(Debug, Clone)]
pub struct Tile {
    /// Tile number in raster order (`tile_row * tile_cols + tile_col`).
    pub number: usize,
    pub data: Vec<u8>,
}

/// A frame to decode: its headers and every tile.
#[derive(Debug, Clone)]
pub struct CodedFrame {
    pub seq: SequenceHeader,
    pub header: FrameHeader,
    pub tiles: Vec<Tile>,
    /// Whether the global motion parameters of each reference (`LAST_FRAME..=ALTREF_FRAME`) pass
    /// the setup shear process (7.11.3.6); true for the types that do not warp.
    pub warp_valid: [bool; 8],
}

/// One frame of a chunk, in decoding order.
#[derive(Debug, Clone)]
pub enum Unit {
    /// `show_existing_frame`: output the frame in reference slot `slot`. When the header's
    /// `frame_type` is a key frame, that frame then refreshes every slot.
    Show { slot: usize, header: Box<FrameHeader> },
    /// Decode this frame; `header.show_frame` says whether it is output, and
    /// `header.refresh_frame_flags` which slots it then occupies.
    Frame(Box<CodedFrame>),
}

struct Pending {
    frame: CodedFrame,
    covered: Vec<bool>,
}

/// The header state of an AV1 stream (see the module docs).
pub struct HeaderParser {
    seq: Option<SequenceHeader>,
    refs: [RefInfo; NUM_REF_FRAMES],
    lf_deltas: LoopFilterDeltas,
    seg_features: crate::header::SegmentationFeatures,
    prev_gm_params: [[i32; 6]; 8],
    current_frame_id: u32,
    pending: Option<Pending>,
    seen_frame_header: bool,
}

impl Default for HeaderParser {
    fn default() -> Self {
        Self::new()
    }
}

impl HeaderParser {
    pub fn new() -> Self {
        Self {
            seq: None,
            refs: Default::default(),
            lf_deltas: LoopFilterDeltas::defaults(),
            seg_features: Default::default(),
            prev_gm_params: default_gm_params(),
            current_frame_id: 0,
            pending: None,
            seen_frame_header: false,
        }
    }

    /// Whether the sequence header among `config_obus` (the `av1C` configuration OBUs) says frames
    /// may carry film grain parameters.
    pub fn announces_film_grain(config_obus: &[u8]) -> bool {
        let mut p = Self::new();
        p.parse(config_obus).is_ok() && p.seq.is_some_and(|s| s.film_grain_params_present)
    }

    /// The active sequence header, once one was parsed.
    pub fn sequence_header(&self) -> Option<&SequenceHeader> {
        self.seq.as_ref()
    }

    /// Parse a chunk of OBUs (low-overhead format) into the frames it completes. Scalable streams
    /// (an operating point that selects layers) are not taken.
    pub fn parse(&mut self, data: &[u8]) -> Result<Vec<Unit>> {
        let mut out = Vec::new();
        let mut pos = 0usize;
        while let Some(&h) = data.get(pos) {
            if h & 0x80 != 0 {
                return Err(Error::Invalid("obu_forbidden_bit"));
            }
            let obu_type = (h >> 3) & 0xf;
            let ext = h & 4 != 0;
            let has_size = h & 2 != 0;
            let mut p = pos + 1;
            let (mut temporal_id, mut spatial_id) = (0, 0);
            if ext {
                let e = *data.get(p).ok_or(Error::Truncated)?;
                temporal_id = u32::from(e >> 5);
                spatial_id = u32::from((e >> 3) & 3);
                p += 1;
            }
            let size = if has_size {
                let (v, n) = leb128(data.get(p..).ok_or(Error::Truncated)?)?;
                p += n;
                usize::try_from(v).map_err(|_| Error::Truncated)?
            } else {
                data.len().checked_sub(p).ok_or(Error::Truncated)?
            };
            let end = p.checked_add(size).filter(|&e| e <= data.len()).ok_or(Error::Truncated)?;
            let payload = data.get(p..end).ok_or(Error::Truncated)?;
            pos = end;
            let unit = match obu_type {
                OBU_SEQUENCE_HEADER => {
                    let s = SequenceHeader::parse(payload)?;
                    if s.op_idc != 0 {
                        return Err(Error::Unsupported("scalability in the hardware path"));
                    }
                    self.seq = Some(s);
                    None
                }
                OBU_TEMPORAL_DELIMITER => {
                    self.seen_frame_header = false;
                    None
                }
                OBU_FRAME_HEADER | OBU_REDUNDANT_FRAME_HEADER | OBU_FRAME => {
                    // frame_header_copy(): identical to the active header
                    if self.seen_frame_header && obu_type != OBU_FRAME {
                        continue;
                    }
                    let mut r = BitReader::new(payload);
                    match self.frame_header(&mut r, temporal_id, spatial_id)? {
                        Some(shown) => Some(shown),
                        None if obu_type == OBU_FRAME => {
                            r.byte_align();
                            let rest = payload.get(r.byte_pos()..).ok_or(Error::Truncated)?;
                            self.tile_group(rest)?
                        }
                        None => None,
                    }
                }
                OBU_TILE_GROUP => self.tile_group(payload)?,
                _ => None,
            };
            if let Some(u) = unit {
                if out.len() >= MAX_UNITS_PER_CHUNK {
                    return Err(Error::Invalid("too many frames in one chunk"));
                }
                out.push(u);
            }
        }
        Ok(out)
    }

    /// frame_header_obu( ): returns the unit of a show_existing_frame.
    fn frame_header(&mut self, r: &mut BitReader, temporal_id: u32, spatial_id: u32) -> Result<Option<Unit>> {
        let seq = self.seq.clone().ok_or(Error::Invalid("frame before sequence header"))?;
        self.seen_frame_header = true;
        let mut st = HeaderState {
            seq: &seq,
            refs: &mut self.refs,
            lf_deltas: self.lf_deltas,
            seg_features: self.seg_features,
            prev_gm_params: self.prev_gm_params,
            current_frame_id: self.current_frame_id,
        };
        let fh = FrameHeader::parse(r, &mut st, temporal_id, spatial_id)?;
        self.lf_deltas = st.lf_deltas;
        self.seg_features = st.seg_features;
        self.prev_gm_params = st.prev_gm_params;
        self.current_frame_id = st.current_frame_id;
        if fh.show_existing_frame {
            self.seen_frame_header = false;
            let slot = fh.frame_to_show_map_idx;
            if fh.frame_type == KEY_FRAME as u8 {
                // reference frame loading process (7.21), then every slot is refreshed (7.20)
                let info = self.refs.get(slot).cloned().ok_or(Error::Invalid("frame_to_show_map_idx"))?;
                self.lf_deltas = info.lf_deltas;
                self.seg_features = info.seg_features;
                for r in self.refs.iter_mut() {
                    *r = info.clone();
                }
            }
            return Ok(Some(Unit::Show { slot, header: Box::new(fh) }));
        }
        let tile_count = fh.tile_info.cols.checked_mul(fh.tile_info.rows).filter(|&n| n > 0).ok_or(Error::Invalid("tile count"))?;
        let mut warp_valid = [true; 8];
        for rf in LAST_FRAME..=ALTREF_FRAME {
            if let (Some(valid), Some(&ty), Some(params)) = (warp_valid.get_mut(rf), fh.gm_type.get(rf), fh.gm_params.get(rf))
                && usize::from(ty) > TRANSLATION
            {
                *valid = setup_shear(params).0;
            }
        }
        self.pending =
            Some(Pending { frame: CodedFrame { seq, header: fh, tiles: Vec::with_capacity(tile_count), warp_valid }, covered: vec![false; tile_count] });
        Ok(None)
    }

    /// tile_group_obu( sz ): returns the frame when its last tile arrived.
    fn tile_group(&mut self, data: &[u8]) -> Result<Option<Unit>> {
        let mut job = self.pending.take().ok_or(Error::Invalid("tile group without a frame header"))?;
        let (cols_log2, rows_log2, tsb) = {
            let t = &job.frame.header.tile_info;
            (t.cols_log2, t.rows_log2, t.tile_size_bytes)
        };
        let num_tiles = job.covered.len();
        let mut r = BitReader::new(data);
        let mut tg_start = 0;
        let mut tg_end = num_tiles.checked_sub(1).ok_or(Error::Invalid("tile count"))?;
        if num_tiles > 1 && r.flag()? {
            let bits = cols_log2 + rows_log2;
            tg_start = r.f(bits)? as usize;
            tg_end = r.f(bits)? as usize;
        }
        r.byte_align();
        if tg_end >= num_tiles || tg_start > tg_end {
            return Err(Error::Invalid("tile group range"));
        }
        let mut pos = r.byte_pos();
        for tile_num in tg_start..=tg_end {
            let size = if tile_num == tg_end {
                data.len().checked_sub(pos).ok_or(Error::Truncated)?
            } else {
                let prefix_end = pos.checked_add(tsb as usize).ok_or(Error::Truncated)?;
                let prefix = data.get(pos..prefix_end).ok_or(Error::Truncated)?;
                let mut v = 0usize;
                for (i, &byte) in prefix.iter().enumerate() {
                    v |= usize::from(byte) << (8 * i);
                }
                pos = prefix_end;
                v.checked_add(1).ok_or(Error::Truncated)?
            };
            let end = pos.checked_add(size).filter(|&e| e <= data.len()).ok_or(Error::Truncated)?;
            let covered = job.covered.get_mut(tile_num).ok_or(Error::Invalid("tile group range"))?;
            if *covered {
                return Err(Error::Invalid("tile is covered by multiple tile groups"));
            }
            *covered = true;
            job.frame.tiles.push(Tile { number: tile_num, data: data.get(pos..end).ok_or(Error::Truncated)?.to_vec() });
            pos = end;
        }
        if tg_end != num_tiles - 1 {
            self.pending = Some(job);
            return Ok(None);
        }
        if job.covered.iter().any(|&c| !c) {
            return Err(Error::Invalid("frame is missing tile coverage"));
        }
        self.seen_frame_header = false;
        // reference frame update process (7.20)
        let fh = &job.frame.header;
        for (i, ri) in self.refs.iter_mut().enumerate() {
            if (fh.refresh_frame_flags >> i) & 1 == 1 {
                ri.valid = true;
                ri.frame_id = fh.current_frame_id;
                ri.upscaled_width = fh.upscaled_width;
                ri.frame_width = fh.frame_width;
                ri.frame_height = fh.frame_height;
                ri.render_width = fh.render_width;
                ri.render_height = fh.render_height;
                ri.mi_cols = fh.mi_cols;
                ri.mi_rows = fh.mi_rows;
                ri.frame_type = fh.frame_type;
                ri.order_hint = fh.order_hint;
                ri.saved_order_hints = fh.order_hints;
                ri.gm_params = fh.gm_params;
                ri.lf_deltas = fh.lf.deltas;
                ri.seg_features = fh.seg.features;
                ri.grain = fh.film_grain.clone();
            }
        }
        Ok(Some(Unit::Frame(Box::new(job.frame))))
    }
}
