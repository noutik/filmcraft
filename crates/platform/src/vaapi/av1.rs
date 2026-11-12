//! The AV1 side of the VA-API decoder, like [`super::h264`] and [`super::hevc`]: the software
//! decoder's own header parser (`filmcraft_av1::hw`: OBUs, uncompressed frame headers, the
//! reference slots' saved state, tile data) drives a stateless hardware decoder, so both decoders
//! make the same decisions and output the same pictures in the same order.
//!
//! What the hardware path does not take is an error, so the hybrid decoder replays the run in
//! software: super-resolution, frames whose size differs from the sequence's (and so references of
//! another size), monochrome and non-4:2:0 pictures, scalable streams, and film grain when the
//! session was made without the surfaces for it.
//!
//! Film grain is applied by the hardware into a second surface (the reference stays clean), kept
//! with the reference slot so a later `show_existing_frame` outputs the same picture.
//!
//! Safe code; compiled on every target so its logic is tested everywhere.

use filmcraft_av1::SequenceHeader;
use filmcraft_av1::hw::{CodedFrame, FrameHeader, HeaderParser, Unit};
use filmcraft_codecs::{CodecError, DecodedFrame, Result};

use super::Accel;
use super::ffi::{
    self, VADecPictureParameterBufferAV1, VAFilmGrainStructAV1, VASegmentationStructAV1, VASliceParameterBufferAV1, VAWarpedMotionParamsAV1, av1_lf_bits,
    av1_lr_bits, av1_mode_bits, av1_pic_bits, av1_qm_bits, av1_seg_bits, av1_seq_bits,
};

/// Reference slots of an AV1 decoder (`NUM_REF_FRAMES`).
const SLOTS: usize = 8;
/// References a frame names (`REFS_PER_FRAME`).
const REFS_PER_FRAME: usize = 7;
/// `KEY_FRAME`.
const KEY_FRAME: u8 = 0;
/// `PRIMARY_REF_NONE`.
const PRIMARY_REF_NONE: usize = 7;
/// `LAST_FRAME`.
const LAST_FRAME: usize = 1;

/// Surfaces an AV1 session needs: every reference slot and the picture being decoded.
pub const SURFACES: usize = SLOTS + 1;
/// Surfaces of a session for a stream with film grain: a display picture besides each of those.
pub const SURFACES_WITH_GRAIN: usize = 2 * SURFACES;

/// What a reference slot holds: the decoded picture and, when film grain was applied to it, the
/// picture to display (both indices into [`Accel::surfaces`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stored {
    surface: usize,
    display: Option<usize>,
}

/// AV1 front end of a stateless hardware decoder (see the module docs).
pub struct Front<A: Accel> {
    accel: A,
    parser: HeaderParser,
    /// The `av1C` configuration OBUs, parsed before the first sample and again after a reset (as
    /// the software decoder is primed).
    config_obus: Vec<u8>,
    primed: bool,
    /// Picture size (luma samples) and bit depth the surfaces were made for.
    size: (u32, u32),
    bits: u32,
    /// What each reference slot holds.
    slots: [Option<Stored>; SLOTS],
}

fn decode_err(e: impl std::fmt::Display) -> CodecError {
    CodecError::Decode(e.to_string())
}

fn unsupported(what: &str) -> CodecError {
    CodecError::Decode(format!("the hardware decoder does not take {what}"))
}

impl<A: Accel> Front<A> {
    /// A front end for a stream whose sample entry carried `config_obus`, decoding into `accel`'s
    /// surfaces of `size` luma samples and `bits` bits.
    pub fn new(accel: A, config_obus: Vec<u8>, size: (u32, u32), bits: u32) -> Self {
        Self { accel, parser: HeaderParser::new(), config_obus, primed: false, size, bits, slots: [None; SLOTS] }
    }

    pub fn accel(&self) -> &A {
        &self.accel
    }

    /// Decode one temporal unit; the pictures it shows come back in order.
    pub fn decode(&mut self, sample: &[u8], pts: i64) -> Result<Vec<DecodedFrame>> {
        if !self.primed {
            self.primed = true;
            if !self.config_obus.is_empty() {
                self.parser.parse(&self.config_obus).map_err(decode_err)?;
            }
        }
        let units = self.parser.parse(sample).map_err(decode_err)?;
        let mut frames = Vec::new();
        for unit in units {
            let shown = match unit {
                Unit::Show { slot, header } => Some(self.show_existing(slot, &header)?),
                Unit::Frame(f) => self.decode_frame(&f)?,
            };
            if let Some(index) = shown {
                let frame = self.accel.read(index).map_err(decode_err)?;
                frames.push(DecodedFrame { pts, frame, draft: false });
            }
        }
        Ok(frames)
    }

    /// Nothing is held back: every picture is returned by the call that shows it.
    pub fn flush(&mut self) -> Vec<DecodedFrame> {
        Vec::new()
    }

    /// Forget every picture (after a seek).
    pub fn reset(&mut self) {
        self.parser = HeaderParser::new();
        self.primed = false;
        self.slots = [None; SLOTS];
    }

    /// `show_existing_frame`: the surface to output.
    fn show_existing(&mut self, slot: usize, header: &FrameHeader) -> Result<usize> {
        let stored = self.slots.get(slot).copied().flatten().ok_or_else(|| decode_err("show_existing_frame of an empty slot"))?;
        // the grain parameters loaded for the frame are the ones it was decoded with (7.21)
        let index = match (header.film_grain.apply_grain, stored.display) {
            (false, _) => stored.surface,
            (true, Some(display)) => display,
            (true, None) => return Err(unsupported("film grain on a frame decoded without it")),
        };
        if header.frame_type == KEY_FRAME {
            // the shown key frame refreshes every slot (7.21)
            self.slots = [Some(stored); SLOTS];
        }
        Ok(index)
    }

    /// A surface no reference slot holds, other than `taken`.
    fn free_surface(&self, taken: Option<usize>) -> Result<usize> {
        let used = |i: usize| taken == Some(i) || self.slots.iter().flatten().any(|s| s.surface == i || s.display == Some(i));
        (0..self.accel.surfaces().len()).find(|&i| !used(i)).ok_or_else(|| unsupported("more pictures than the decoder's surfaces"))
    }

    /// Decode `f` into a free surface and store it in the slots it refreshes; the surface to
    /// output when the frame is shown.
    fn decode_frame(&mut self, f: &CodedFrame) -> Result<Option<usize>> {
        let (seq, fh) = (&f.seq, &f.header);
        self.check_supported(seq, fh)?;
        let target = self.free_surface(None)?;
        let display = if fh.film_grain.apply_grain { Some(self.free_surface(Some(target))?) } else { None };
        let surfaces = self.accel.surfaces();
        let id_of = |i: usize| surfaces.get(i).copied().ok_or_else(|| decode_err("surface out of range"));
        let target_id = id_of(target)?;
        let mut pic = picture_params(seq, fh, &f.warp_valid)?;
        pic.current_frame = target_id;
        pic.current_display_picture = id_of(display.unwrap_or(target))?;
        for (id, slot) in pic.ref_frame_map.iter_mut().zip(&self.slots) {
            // A slot nothing was stored in is not used by this frame (checked below for the ones
            // it names); the driver still wants a valid surface there.
            *id = match slot {
                Some(s) => id_of(s.surface)?,
                None if fh.frame_is_intra => ffi::VA_INVALID_SURFACE,
                None => target_id,
            };
        }
        if !fh.frame_is_intra {
            for &idx in &fh.ref_frame_idx {
                if self.slots.get(idx).copied().flatten().is_none() {
                    return Err(decode_err("missing reference frame"));
                }
            }
        }
        if fh.primary_ref_frame != PRIMARY_REF_NONE {
            let idx = fh.ref_frame_idx.get(fh.primary_ref_frame).copied().ok_or_else(|| decode_err("primary_ref_frame"))?;
            if self.slots.get(idx).copied().flatten().is_none() {
                return Err(decode_err("primary reference frame missing"));
            }
        }
        let tiles = tile_params(fh, f)?;
        self.accel.decode_av1(target, &pic, &tiles).map_err(decode_err)?;
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if (fh.refresh_frame_flags >> i) & 1 == 1 {
                *slot = Some(Stored { surface: target, display });
            }
        }
        Ok(fh.show_frame.then_some(display.unwrap_or(target)))
    }

    fn check_supported(&self, seq: &SequenceHeader, fh: &FrameHeader) -> Result<()> {
        let c = &seq.color;
        if seq.profile != 0 || c.mono_chrome || (c.subsampling_x, c.subsampling_y) != (1, 1) || u32::from(c.bit_depth) != self.bits {
            return Err(unsupported("a picture format change"));
        }
        if (fh.upscaled_width, fh.frame_height) != self.size {
            return Err(unsupported("a picture size change"));
        }
        if fh.use_superres || fh.frame_width != fh.upscaled_width {
            return Err(unsupported("super-resolution"));
        }
        if fh.film_grain.apply_grain && self.accel.surfaces().len() < SURFACES_WITH_GRAIN {
            return Err(unsupported("film grain in a stream that did not announce it"));
        }
        Ok(())
    }
}

fn flag(on: bool, bit: u32) -> u32 {
    u32::from(on) << bit
}

fn u8_of(v: u32) -> u8 {
    u8::try_from(v).unwrap_or(u8::MAX)
}

fn i8_of(v: i32) -> i8 {
    i8::try_from(v).unwrap_or(0)
}

fn slot_u8(v: usize) -> u8 {
    u8::try_from(v).unwrap_or(0)
}

/// Tile sizes minus 1 in superblocks from the tile start positions (mode info units; the last entry
/// is the frame's end). The last tile's size is left to the driver, as the buffer has no room for it.
fn tile_sizes(starts: &[u32], sb_mi: u32, out: &mut [u16; 63]) -> Result<()> {
    let tiles = starts.len().saturating_sub(1);
    if tiles == 0 || tiles > out.len() + 1 {
        return Err(unsupported("that many tiles"));
    }
    for (o, w) in out.iter_mut().zip(starts.windows(2)) {
        let mi = w.get(1).copied().unwrap_or(0).saturating_sub(w.first().copied().unwrap_or(0));
        *o = u16::try_from(mi.div_ceil(sb_mi.max(1)).saturating_sub(1)).unwrap_or(0);
    }
    Ok(())
}

fn segmentation(fh: &FrameHeader) -> VASegmentationStructAV1 {
    use av1_seg_bits as b;
    let s = &fh.seg;
    let mut out = VASegmentationStructAV1 {
        segment_info_fields: flag(s.enabled, b::ENABLED)
            | flag(s.update_map, b::UPDATE_MAP)
            | flag(s.temporal_update, b::TEMPORAL_UPDATE)
            | flag(s.update_data, b::UPDATE_DATA),
        ..Default::default()
    };
    for (i, (data, mask)) in out.feature_data.iter_mut().zip(out.feature_mask.iter_mut()).enumerate() {
        let (Some(enabled), Some(values)) = (s.features.enabled.get(i), s.features.data.get(i)) else { continue };
        for (j, d) in data.iter_mut().enumerate() {
            if enabled.get(j).copied().unwrap_or(false) {
                *mask |= 1 << j;
                *d = i16::try_from(values.get(j).copied().unwrap_or(0)).unwrap_or(0);
            }
        }
    }
    out
}

fn film_grain(fh: &FrameHeader) -> VAFilmGrainStructAV1 {
    use ffi::av1_grain_bits as b;
    let g = &fh.film_grain;
    let mut out = VAFilmGrainStructAV1::default();
    if !g.apply_grain {
        return out;
    }
    out.film_grain_info_fields = flag(true, b::APPLY_GRAIN)
        | flag(g.chroma_scaling_from_luma, b::CHROMA_SCALING_FROM_LUMA)
        | (g.grain_scaling_minus_8 & 3) << b::GRAIN_SCALING_MINUS_8
        | (g.ar_coeff_lag & 3) << b::AR_COEFF_LAG
        | (g.ar_coeff_shift_minus_6 & 3) << b::AR_COEFF_SHIFT_MINUS_6
        | (g.grain_scale_shift & 3) << b::GRAIN_SCALE_SHIFT
        | flag(g.overlap_flag, b::OVERLAP_FLAG)
        | flag(g.clip_to_restricted_range, b::CLIP_TO_RESTRICTED_RANGE);
    out.grain_seed = u16::try_from(g.grain_seed).unwrap_or(0);
    let count = |n: usize, max: usize| u8::try_from(n.min(max)).unwrap_or(0);
    out.num_y_points = count(g.num_y_points, 14);
    out.num_cb_points = count(g.num_cb_points, 10);
    out.num_cr_points = count(g.num_cr_points, 10);
    let copy = |dst: &mut [u8], src: &[u8]| {
        for (d, s) in dst.iter_mut().zip(src) {
            *d = *s;
        }
    };
    copy(&mut out.point_y_value, &g.point_y_value);
    copy(&mut out.point_y_scaling, &g.point_y_scaling);
    copy(&mut out.point_cb_value, &g.point_cb_value);
    copy(&mut out.point_cb_scaling, &g.point_cb_scaling);
    copy(&mut out.point_cr_value, &g.point_cr_value);
    copy(&mut out.point_cr_scaling, &g.point_cr_scaling);
    let signed = |dst: &mut [i8], src: &[u8]| {
        for (d, s) in dst.iter_mut().zip(src) {
            *d = i8_of(i32::from(*s) - 128);
        }
    };
    signed(&mut out.ar_coeffs_y, &g.ar_coeffs_y_plus_128);
    signed(&mut out.ar_coeffs_cb, &g.ar_coeffs_cb_plus_128);
    signed(&mut out.ar_coeffs_cr, &g.ar_coeffs_cr_plus_128);
    out.cb_mult = u8_of(g.cb_mult);
    out.cb_luma_mult = u8_of(g.cb_luma_mult);
    out.cb_offset = u16::try_from(g.cb_offset).unwrap_or(0);
    out.cr_mult = u8_of(g.cr_mult);
    out.cr_luma_mult = u8_of(g.cr_luma_mult);
    out.cr_offset = u16::try_from(g.cr_offset).unwrap_or(0);
    out
}

/// A CDEF strength as the header codes it: the primary strength above the 2-bit secondary one
/// (whose coded value 3 means 4).
fn cdef_strength(pri: u32, sec: u32) -> u8 {
    let sec = if sec == 4 { 3 } else { sec & 3 };
    u8_of((pri & 0xf) << 2 | sec)
}

fn loop_restoration(fh: &FrameHeader) -> u16 {
    use av1_lr_bits as b;
    let l = &fh.lr;
    let ty = |plane: usize| u32::from(l.frame_restoration_type.get(plane).copied().unwrap_or(0) & 3);
    let mut v = ty(0) << b::YFRAME_RESTORATION_TYPE | ty(1) << b::CBFRAME_RESTORATION_TYPE | ty(2) << b::CRFRAME_RESTORATION_TYPE;
    if l.uses_lr {
        // LoopRestorationSize[ 0 ] = 64 << lr_unit_shift; chroma units are that or half of it
        let [luma, chroma, _] = l.loop_restoration_size;
        let shift = luma.max(64).ilog2().saturating_sub(6).min(2);
        v |= shift << b::LR_UNIT_SHIFT | flag(chroma != luma, b::LR_UV_SHIFT);
    }
    u16::try_from(v).unwrap_or(0)
}

/// The picture parameters of frame `fh` of sequence `seq` (the current picture, its display
/// picture and the reference surfaces are filled in by the caller).
pub fn picture_params(seq: &SequenceHeader, fh: &FrameHeader, warp_valid: &[bool; 8]) -> Result<VADecPictureParameterBufferAV1> {
    use {av1_lf_bits as lf, av1_mode_bits as m, av1_pic_bits as p, av1_qm_bits as qm, av1_seq_bits as s};
    let c = &seq.color;
    let seq_info_fields = flag(seq.still_picture, s::STILL_PICTURE)
        | flag(seq.use_128x128_superblock, s::USE_128X128_SUPERBLOCK)
        | flag(seq.enable_filter_intra, s::ENABLE_FILTER_INTRA)
        | flag(seq.enable_intra_edge_filter, s::ENABLE_INTRA_EDGE_FILTER)
        | flag(seq.enable_interintra_compound, s::ENABLE_INTERINTRA_COMPOUND)
        | flag(seq.enable_masked_compound, s::ENABLE_MASKED_COMPOUND)
        | flag(seq.enable_dual_filter, s::ENABLE_DUAL_FILTER)
        | flag(seq.enable_order_hint, s::ENABLE_ORDER_HINT)
        | flag(seq.enable_jnt_comp, s::ENABLE_JNT_COMP)
        | flag(seq.enable_cdef, s::ENABLE_CDEF)
        | flag(c.mono_chrome, s::MONO_CHROME)
        | flag(c.color_range, s::COLOR_RANGE)
        | flag(c.subsampling_x != 0, s::SUBSAMPLING_X)
        | flag(c.subsampling_y != 0, s::SUBSAMPLING_Y)
        | flag(seq.film_grain_params_present, s::FILM_GRAIN_PARAMS_PRESENT);
    let t = &fh.tile_info;
    let pic_info_fields = u32::from(fh.frame_type & 3) << p::FRAME_TYPE
        | flag(fh.show_frame, p::SHOW_FRAME)
        | flag(fh.showable_frame, p::SHOWABLE_FRAME)
        | flag(fh.error_resilient_mode, p::ERROR_RESILIENT_MODE)
        | flag(fh.disable_cdf_update, p::DISABLE_CDF_UPDATE)
        | flag(fh.allow_screen_content_tools, p::ALLOW_SCREEN_CONTENT_TOOLS)
        | flag(fh.force_integer_mv, p::FORCE_INTEGER_MV)
        | flag(fh.allow_intrabc, p::ALLOW_INTRABC)
        | flag(fh.use_superres, p::USE_SUPERRES)
        | flag(fh.allow_high_precision_mv, p::ALLOW_HIGH_PRECISION_MV)
        | flag(fh.is_motion_mode_switchable, p::IS_MOTION_MODE_SWITCHABLE)
        | flag(fh.use_ref_frame_mvs, p::USE_REF_FRAME_MVS)
        | flag(fh.disable_frame_end_update_cdf, p::DISABLE_FRAME_END_UPDATE_CDF)
        | flag(t.uniform, p::UNIFORM_TILE_SPACING)
        | flag(fh.allow_warped_motion, p::ALLOW_WARPED_MOTION);
    let loop_filter_info_fields = (fh.lf.sharpness & 7) << lf::SHARPNESS_LEVEL
        | flag(fh.lf.delta_enabled, lf::MODE_REF_DELTA_ENABLED)
        | flag(fh.lf.delta_update, lf::MODE_REF_DELTA_UPDATE);
    let q = &fh.quant;
    let qmatrix_fields = flag(q.using_qmatrix, qm::USING_QMATRIX) | (q.qm_y & 0xf) << qm::QM_Y | (q.qm_u & 0xf) << qm::QM_U | (q.qm_v & 0xf) << qm::QM_V;
    let mode_control_fields = flag(fh.delta_q_present, m::DELTA_Q_PRESENT)
        | (fh.delta_q_res & 3) << m::LOG2_DELTA_Q_RES
        | flag(fh.delta_lf_present, m::DELTA_LF_PRESENT)
        | (fh.delta_lf_res & 3) << m::LOG2_DELTA_LF_RES
        | flag(fh.delta_lf_multi, m::DELTA_LF_MULTI)
        | u32::from(fh.tx_mode & 3) << m::TX_MODE
        | flag(fh.reference_select, m::REFERENCE_SELECT)
        | flag(fh.reduced_tx_set, m::REDUCED_TX_SET_USED)
        | flag(fh.skip_mode_present, m::SKIP_MODE_PRESENT);
    let narrow = |v: u32| u16::try_from(v.saturating_sub(1)).map_err(|_| decode_err("picture too large"));
    let sb_mi = if seq.use_128x128_superblock { 32 } else { 16 };
    let mut width_in_sbs_minus_1 = [0u16; 63];
    let mut height_in_sbs_minus_1 = [0u16; 63];
    tile_sizes(&t.mi_col_starts, sb_mi, &mut width_in_sbs_minus_1)?;
    tile_sizes(&t.mi_row_starts, sb_mi, &mut height_in_sbs_minus_1)?;
    let mut ref_frame_idx = [0u8; REFS_PER_FRAME];
    for (o, &i) in ref_frame_idx.iter_mut().zip(&fh.ref_frame_idx) {
        *o = slot_u8(i);
    }
    let mut ref_deltas = [0i8; 8];
    for (o, &d) in ref_deltas.iter_mut().zip(&fh.lf.deltas.ref_deltas) {
        *o = i8_of(d);
    }
    let mut cdef_y_strengths = [0u8; 8];
    let mut cdef_uv_strengths = [0u8; 8];
    for i in 0..8 {
        cdef_y_strengths[i] = cdef_strength(fh.cdef.y_pri[i], fh.cdef.y_sec[i]);
        cdef_uv_strengths[i] = cdef_strength(fh.cdef.uv_pri[i], fh.cdef.uv_sec[i]);
    }
    let mut wm = [VAWarpedMotionParamsAV1::default(); REFS_PER_FRAME];
    for (i, w) in wm.iter_mut().enumerate() {
        let rf = LAST_FRAME + i;
        w.wmtype = i32::from(fh.gm_type.get(rf).copied().unwrap_or(0));
        if let Some(params) = fh.gm_params.get(rf) {
            for (o, &v) in w.wmmat.iter_mut().zip(params) {
                *o = v;
            }
        }
        w.invalid = u8::from(!warp_valid.get(rf).copied().unwrap_or(true));
    }
    Ok(VADecPictureParameterBufferAV1 {
        profile: seq.profile,
        order_hint_bits_minus_1: u8_of(seq.order_hint_bits.saturating_sub(1)),
        bit_depth_idx: c.bit_depth.saturating_sub(8) / 2,
        matrix_coefficients: c.matrix_coefficients,
        seq_info_fields,
        current_frame: ffi::VA_INVALID_SURFACE,
        current_display_picture: ffi::VA_INVALID_SURFACE,
        anchor_frames_num: 0,
        anchor_frames_list: 0,
        frame_width_minus1: narrow(fh.upscaled_width)?,
        frame_height_minus1: narrow(fh.frame_height)?,
        output_frame_width_in_tiles_minus_1: 0,
        output_frame_height_in_tiles_minus_1: 0,
        ref_frame_map: [ffi::VA_INVALID_SURFACE; 8],
        ref_frame_idx,
        primary_ref_frame: slot_u8(fh.primary_ref_frame),
        order_hint: u8_of(fh.order_hint),
        seg_info: segmentation(fh),
        film_grain_info: film_grain(fh),
        tile_cols: u8::try_from(t.cols).map_err(|_| unsupported("that many tiles"))?,
        tile_rows: u8::try_from(t.rows).map_err(|_| unsupported("that many tiles"))?,
        width_in_sbs_minus_1,
        height_in_sbs_minus_1,
        tile_count_minus_1: 0,
        context_update_tile_id: u16::try_from(t.context_update_tile_id).unwrap_or(0),
        pic_info_fields,
        superres_scale_denominator: u8_of(fh.superres_denom),
        interp_filter: fh.interpolation_filter,
        filter_level: [u8_of(fh.lf.level[0]), u8_of(fh.lf.level[1])],
        filter_level_u: u8_of(fh.lf.level[2]),
        filter_level_v: u8_of(fh.lf.level[3]),
        loop_filter_info_fields: u8_of(loop_filter_info_fields),
        ref_deltas,
        mode_deltas: [i8_of(fh.lf.deltas.mode_deltas[0]), i8_of(fh.lf.deltas.mode_deltas[1])],
        base_qindex: u8_of(q.base_q_idx),
        y_dc_delta_q: i8_of(q.delta_q_y_dc),
        u_dc_delta_q: i8_of(q.delta_q_u_dc),
        u_ac_delta_q: i8_of(q.delta_q_u_ac),
        v_dc_delta_q: i8_of(q.delta_q_v_dc),
        v_ac_delta_q: i8_of(q.delta_q_v_ac),
        qmatrix_fields: u16::try_from(qmatrix_fields).unwrap_or(0),
        mode_control_fields,
        cdef_damping_minus_3: u8_of(fh.cdef.damping.saturating_sub(3)),
        cdef_bits: u8_of(fh.cdef.bits),
        cdef_y_strengths,
        cdef_uv_strengths,
        loop_restoration_fields: loop_restoration(fh),
        wm,
        va_reserved: [0; ffi::VA_PADDING_MEDIUM],
    })
}

/// The tiles of `f` as the hardware takes them: one parameter and data buffer each.
fn tile_params(fh: &FrameHeader, f: &CodedFrame) -> Result<Vec<(VASliceParameterBufferAV1, Vec<u8>)>> {
    let cols = fh.tile_info.cols.max(1);
    let last = u16::try_from(f.tiles.len().saturating_sub(1)).map_err(|_| unsupported("that many tiles"))?;
    f.tiles
        .iter()
        .map(|t| {
            let params = VASliceParameterBufferAV1 {
                slice_data_size: u32::try_from(t.data.len()).map_err(|_| decode_err("tile too large"))?,
                slice_data_offset: 0,
                slice_data_flag: ffi::VA_SLICE_DATA_FLAG_ALL,
                tile_row: u16::try_from(t.number / cols).map_err(|_| decode_err("tile number"))?,
                tile_column: u16::try_from(t.number % cols).map_err(|_| decode_err("tile number"))?,
                tg_start: 0,
                tg_end: last,
                ..Default::default()
            };
            Ok((params, t.data.clone()))
        })
        .collect()
}
