pub(crate) fn encode_pgm(width: u32, height: u32, data: &[u8]) -> Vec<u8> {
    let mut out = format!("P5\n{width} {height}\n255\n").into_bytes();
    out.extend_from_slice(data);
    out
}

fn skip_whitespace_and_comments(bytes: &[u8], pos: &mut usize) {
    loop {
        while *pos < bytes.len() && bytes[*pos].is_ascii_whitespace() {
            *pos += 1;
        }
        if *pos < bytes.len() && bytes[*pos] == b'#' {
            while *pos < bytes.len() && bytes[*pos] != b'\n' {
                *pos += 1;
            }
        } else {
            break;
        }
    }
}

fn read_token<'a>(bytes: &'a [u8], pos: &mut usize) -> Result<&'a [u8], String> {
    skip_whitespace_and_comments(bytes, pos);
    let start = *pos;
    while *pos < bytes.len() && !bytes[*pos].is_ascii_whitespace() {
        *pos += 1;
    }
    if *pos == start {
        return Err("unexpected end of PGM header".to_string());
    }
    Ok(&bytes[start..*pos])
}

fn parse_u32(token: &[u8], what: &str) -> Result<u32, String> {
    std::str::from_utf8(token)
        .ok()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("invalid {what}"))
}

/// Parses a binary PGM (`P5`) image: magic, whitespace/`#`-comment-separated width, height and
/// maxval (must be 255), exactly one whitespace byte, then exactly `width * height` raster bytes.
pub(crate) fn decode_pgm(bytes: &[u8]) -> Result<(u32, u32, &[u8]), String> {
    if bytes.len() < 2 || &bytes[..2] != b"P5" {
        return Err("not a P5 PGM".to_string());
    }
    let mut pos = 2;
    let width = parse_u32(read_token(bytes, &mut pos)?, "width")?;
    let height = parse_u32(read_token(bytes, &mut pos)?, "height")?;
    let maxval = parse_u32(read_token(bytes, &mut pos)?, "maxval")?;
    if maxval != 255 {
        return Err(format!("unsupported maxval {maxval}"));
    }
    match bytes.get(pos) {
        Some(b) if b.is_ascii_whitespace() => pos += 1,
        _ => return Err("missing whitespace after maxval".to_string()),
    }
    let raster = &bytes[pos..];
    let expected = width as usize * height as usize;
    if raster.len() != expected {
        return Err(format!(
            "expected {expected} raster bytes, got {}",
            raster.len()
        ));
    }
    Ok((width, height, raster))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_pgm_header_and_payload() {
        let data = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let encoded = encode_pgm(4, 2, &data);
        assert_eq!(&encoded[..11], b"P5\n4 2\n255\n");
        assert_eq!(&encoded[11..], &data);
    }

    #[test]
    fn test_decode_pgm_roundtrips_encode() {
        let px = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let encoded = encode_pgm(4, 2, &px);
        assert_eq!(decode_pgm(&encoded), Ok((4, 2, &px[..])));
    }

    #[test]
    fn test_decode_pgm_accepts_comments() {
        let mut bytes = b"P5\n# eye\n4 2\n255\n".to_vec();
        bytes.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(
            decode_pgm(&bytes),
            Ok((4, 2, &[1, 2, 3, 4, 5, 6, 7, 8][..]))
        );
    }

    #[test]
    fn test_decode_pgm_rejects_bad_input() {
        assert!(decode_pgm(b"P6\n4 2\n255\n\x01\x02\x03\x04\x05\x06\x07\x08").is_err());
        assert!(decode_pgm(b"P5\n4 2\n65535\n\x01\x02\x03\x04\x05\x06\x07\x08").is_err());
        assert!(decode_pgm(b"P5\n4 2\n255\n\x01\x02\x03").is_err());
    }
}
