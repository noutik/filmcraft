//! Diagnostics for the VA-API backend (Linux): does it take a stream, do its pictures equal the
//! software decoder's, and how fast is each? `vaprobe [--frames N] [--time] <file.mp4>...`
//! (`--time` skips the comparison and only times the hardware decoder).
#[cfg(target_os = "linux")]
fn main() {
    use std::collections::BTreeMap;

    use filmcraft_codecs::{DecodedFrame, VideoDecoder};
    use filmcraft_frame::PixelData;

    println!("register: {:?}", filmcraft_platform::register());
    let args: Vec<String> = std::env::args().skip(1).collect();
    let time_only = args.iter().any(|a| a == "--time");
    let frames = args.iter().position(|a| a == "--frames").and_then(|i| args.get(i + 1)).and_then(|n| n.parse::<usize>().ok()).unwrap_or(120);
    let mut skip = false;
    for path in &args {
        if skip {
            skip = false;
            continue;
        }
        if path == "--frames" {
            skip = true;
            continue;
        }
        if path.starts_with("--") {
            continue;
        }
        let Ok(bytes) = std::fs::read(path) else {
            println!("{path}: unreadable");
            continue;
        };
        let Ok(file) = filmcraft_isobmff::open(bytes.clone()) else {
            println!("{path}: not an MP4");
            continue;
        };
        let Some(t) = file.track_of_kind(filmcraft_isobmff::TrackKind::Video) else {
            println!("{path}: no video");
            continue;
        };
        let Some(entry) = file.tracks[t].entries.first() else { continue };
        let n = file.tracks[t].samples.len().min(frames);
        let samples: Vec<(Vec<u8>, i64)> = (0..n).filter_map(|i| Some((file.read_sample(&bytes, t, i).ok()?, file.tracks[t].samples[i].pts))).collect();
        let Some(Ok(mut hw)) = filmcraft_platform::vaapi_factory(entry) else {
            println!("{path}: no hardware decoder for this stream");
            continue;
        };
        // Each picture is kept as a summary (size, colour, hash of the samples), not as pixels:
        // a long 1440p file would need tens of gigabytes for the pictures of both decoders.
        let summary = |f: &DecodedFrame| -> (u32, u32, String, u64) {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            match &f.frame.data {
                PixelData::Yuv8 { planes, .. } => planes.iter().for_each(|p| p.iter().for_each(|&v| u16::from(v).hash(&mut h))),
                PixelData::Yuv16 { planes, .. } => planes.iter().for_each(|p| p.hash(&mut h)),
                _ => 0u8.hash(&mut h),
            }
            (f.frame.width, f.frame.height, format!("{:?}", f.frame.color), h.finish())
        };
        let run = |d: &mut dyn VideoDecoder| -> (BTreeMap<i64, (u32, u32, String, u64)>, f64, Option<String>) {
            let t0 = std::time::Instant::now();
            let mut out = BTreeMap::new();
            let mut err = None;
            for (s, pts) in &samples {
                match d.decode(s, *pts) {
                    Ok(fs) => out.extend(fs.iter().map(|f| (f.pts, summary(f)))),
                    Err(e) => {
                        err = Some(e.to_string());
                        break;
                    }
                }
            }
            out.extend(d.flush().iter().map(|f| (f.pts, summary(f))));
            (out, t0.elapsed().as_secs_f64(), err)
        };
        let (hw_out, hw_secs, hw_err) = run(hw.as_mut());
        println!(
            "{path}: {} -> {} pictures in {hw_secs:.2} s ({:.0} fps){}",
            hw.name(),
            hw_out.len(),
            hw_out.len() as f64 / hw_secs.max(1e-9),
            hw_err.map(|e| format!(", error: {e}")).unwrap_or_default()
        );
        if time_only {
            continue;
        }
        let Ok(mut sw) = filmcraft_codecs::software_video_decoder(entry) else { continue };
        let (sw_out, sw_secs, sw_err) = run(sw.as_mut());
        println!(
            "  {} -> {} pictures in {sw_secs:.2} s ({:.0} fps){}",
            sw.name(),
            sw_out.len(),
            sw_out.len() as f64 / sw_secs.max(1e-9),
            sw_err.map(|e| format!(", error: {e}")).unwrap_or_default()
        );
        let (mut equal, mut differ) = (0usize, 0usize);
        for (pts, h) in &hw_out {
            let Some(s) = sw_out.get(pts) else {
                println!("  pts {pts}: only from the hardware decoder");
                continue;
            };
            if h == s {
                equal += 1;
            } else {
                differ += 1;
                if differ <= 5 {
                    println!("  pts {pts}: {}x{} {} vs {}x{} {}{}", h.0, h.1, h.2, s.0, s.1, s.2, if h.3 == s.3 { "" } else { ", samples differ" });
                }
            }
        }
        println!("  identical pictures: {equal}, different: {differ}, software-only: {}", sw_out.keys().filter(|k| !hw_out.contains_key(k)).count());
    }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    println!("vaprobe is for Linux (VA-API)");
}
