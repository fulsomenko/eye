use std::borrow::Cow;

use eye_core::image::GrayImage;
use zune_jpeg::{
    JpegDecoder,
    zune_core::{colorspace::ColorSpace, options::DecoderOptions},
};

use crate::{DetectError, image::RgbImage};

const APP0_AVI1: [u8; 9] = [0xFF, 0xE0, 0x00, 0x07, b'A', b'V', b'I', b'1', 0x00];

pub fn ensure_huffman_tables(bytes: &[u8]) -> Result<Cow<'_, [u8]>, DetectError> {
    if bytes.len() < 4 || bytes[..2] != [0xFF, 0xD8] {
        return Err(DetectError::Decode("not a JPEG (no SOI)".into()));
    }
    let mut i = 2;
    while i + 1 < bytes.len() && bytes[i] == 0xFF {
        match bytes[i + 1] {
            0xFF => i += 1,
            0xC4 => return Ok(Cow::Borrowed(bytes)),
            0xDA => break,
            0x01 | 0xD0..=0xD7 => i += 2,
            _ => {
                let Some(&[hi, lo]) = bytes.get(i + 2..i + 4) else {
                    break;
                };
                let len = usize::from(u16::from_be_bytes([hi, lo]));
                if len < 2 || i + 2 + len > bytes.len() {
                    break;
                }
                i += 2 + len;
            }
        }
    }
    let mut out = Vec::with_capacity(bytes.len() + APP0_AVI1.len());
    out.extend_from_slice(&bytes[..2]);
    out.extend_from_slice(&APP0_AVI1);
    out.extend_from_slice(&bytes[2..]);
    Ok(Cow::Owned(out))
}

fn decode(bytes: &[u8], colorspace: ColorSpace) -> Result<(u32, u32, Vec<u8>), DetectError> {
    let stream = ensure_huffman_tables(bytes)?;
    let options = DecoderOptions::default().jpeg_set_out_colorspace(colorspace);
    let mut decoder = JpegDecoder::new_with_options(stream.as_ref(), options);
    let data = decoder
        .decode()
        .map_err(|e| DetectError::Decode(e.to_string()))?;
    let (w, h) = decoder
        .dimensions()
        .ok_or_else(|| DetectError::Decode("no frame header".into()))?;
    let dim =
        |v: usize| u32::try_from(v).map_err(|_| DetectError::Decode("dimension overflow".into()));
    Ok((dim(w)?, dim(h)?, data))
}

pub fn decode_mjpeg_gray(bytes: &[u8]) -> Result<GrayImage, DetectError> {
    let (w, h, data) = decode(bytes, ColorSpace::Luma)?;
    Ok(GrayImage::new(w, h, data)?)
}

pub fn decode_mjpeg_rgb(bytes: &[u8]) -> Result<RgbImage<'static>, DetectError> {
    let (width, height, data) = decode(bytes, ColorSpace::RGB)?;
    Ok(RgbImage::owned(width, height, data))
}

#[cfg(test)]
mod tests {
    use image::{ExtendedColorType, codecs::jpeg::JpegEncoder};

    use super::*;

    fn encode_jpeg(data: &[u8], width: u32, height: u32, color: ExtendedColorType) -> Vec<u8> {
        let mut out = Vec::new();
        JpegEncoder::new_with_quality(&mut out, 95)
            .encode(data, width, height, color)
            .unwrap();
        out
    }

    fn strip_dht(bytes: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(bytes.len());
        out.extend_from_slice(&bytes[..2]);
        let mut i = 2;
        while i + 1 < bytes.len() {
            if bytes[i] != 0xFF {
                break;
            }
            let marker = bytes[i + 1];
            if marker == 0xDA {
                out.extend_from_slice(&bytes[i..]);
                break;
            }
            let len = usize::from(u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]));
            if marker != 0xC4 {
                out.extend_from_slice(&bytes[i..i + 2 + len]);
            }
            i += 2 + len;
        }
        out
    }

    fn red_blue_rgb(width: u32, height: u32) -> Vec<u8> {
        let mut data = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                let _ = y;
                if x < width / 2 {
                    data.extend_from_slice(&[255, 0, 0]);
                } else {
                    data.extend_from_slice(&[0, 0, 255]);
                }
            }
        }
        data
    }

    #[test]
    fn test_decode_mjpeg_without_dht_succeeds_after_splice() {
        let original = red_blue_rgb(32, 32);
        let jpeg = encode_jpeg(&original, 32, 32, ExtendedColorType::Rgb8);
        let stripped = strip_dht(&jpeg);

        assert!(JpegDecoder::new(stripped.as_slice()).decode().is_err());

        let decoded_original = decode_mjpeg_rgb(&jpeg).unwrap();
        let decoded_stripped = decode_mjpeg_rgb(&stripped).unwrap();
        assert_eq!(decoded_original.data.len(), decoded_stripped.data.len());
        for (a, b) in decoded_original
            .data
            .iter()
            .zip(decoded_stripped.data.iter())
        {
            assert!((i16::from(*a) - i16::from(*b)).abs() <= 6);
        }
    }

    #[test]
    fn test_decode_rgb_jpeg_returns_interleaved_rgb() {
        let data = red_blue_rgb(32, 32);
        let jpeg = encode_jpeg(&data, 32, 32, ExtendedColorType::Rgb8);
        let decoded = decode_mjpeg_rgb(&jpeg).unwrap();
        assert_eq!(decoded.data.len(), 3072);
        assert!(decoded.get(4, 16, 0) > 200 && decoded.get(4, 16, 2) < 60);
        assert!(decoded.get(28, 16, 2) > 200 && decoded.get(28, 16, 0) < 60);
    }

    #[test]
    fn test_decode_gray_jpeg_matches_source_within_quantization() {
        let width = 64u32;
        let height = 48u32;
        let mut data = Vec::with_capacity((width * height) as usize);
        for i in 0..(width * height) {
            data.push(((i % width) * 4) as u8);
        }
        let jpeg = encode_jpeg(&data, width, height, ExtendedColorType::L8);
        let decoded = decode_mjpeg_gray(&jpeg).unwrap();
        assert_eq!(decoded.width(), width);
        assert_eq!(decoded.height(), height);
        for (src, dst) in data.iter().zip(decoded.data().iter()) {
            assert!((i16::from(*src) - i16::from(*dst)).abs() <= 6);
        }
    }

    #[test]
    fn test_ensure_huffman_tables_borrows_when_dht_present() {
        let data = vec![0u8; 100];
        let jpeg = encode_jpeg(&data, 10, 10, ExtendedColorType::L8);
        match ensure_huffman_tables(&jpeg).unwrap() {
            Cow::Borrowed(_) => {}
            Cow::Owned(_) => panic!("expected borrowed"),
        }
    }

    #[test]
    fn test_decode_rejects_non_jpeg_with_decode_error() {
        assert!(matches!(
            decode_mjpeg_gray(&[0, 1, 2]),
            Err(DetectError::Decode(_))
        ));
        assert!(matches!(
            decode_mjpeg_gray(&[]),
            Err(DetectError::Decode(_))
        ));
        assert!(matches!(
            decode_mjpeg_gray(&[0xFF, 0xD8]),
            Err(DetectError::Decode(_))
        ));
    }

    #[test]
    fn test_ensure_huffman_tables_truncated_segment_splices_without_panic() {
        let bytes = [0xFFu8, 0xD8, 0xFF, 0xE0, 0x00, 0x40, 0x01];
        let result = ensure_huffman_tables(&bytes).unwrap();
        match result {
            Cow::Owned(v) => assert_eq!(v.len(), 16),
            Cow::Borrowed(_) => panic!("expected owned"),
        }
        assert!(matches!(
            decode_mjpeg_gray(&bytes),
            Err(DetectError::Decode(_))
        ));
    }

    #[test]
    #[ignore = "needs hardware"]
    fn test_decode_real_mjpg_frame() {
        let path = std::env::var("EYE_MJPG_FRAME").expect("EYE_MJPG_FRAME not set");
        let bytes = std::fs::read(path).expect("failed to read EYE_MJPG_FRAME");

        let decoded = decode_mjpeg_rgb(&bytes).unwrap();
        assert_eq!(decoded.width, 1280);
        assert_eq!(decoded.height, 720);

        match ensure_huffman_tables(&bytes).unwrap() {
            Cow::Borrowed(_) => println!("ensure_huffman_tables: Borrowed (DHT present)"),
            Cow::Owned(_) => println!("ensure_huffman_tables: Owned (DHT absent, spliced)"),
        }
    }
}
