pub(crate) fn encode_pgm(width: u32, height: u32, data: &[u8]) -> Vec<u8> {
    let mut out = format!("P5\n{width} {height}\n255\n").into_bytes();
    out.extend_from_slice(data);
    out
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
}
