use eye_core::PixelFormat;
use v4l::FourCC;

/// A pixel format the V4L2 source can capture and the session writer can store.
/// Serde names match `PixelFormat`'s: `[[camera]] format = "gray"` and `session.toml`
/// `format = "gray"` are the same spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum StoredFormat {
    #[serde(rename = "gray")]
    Gray8,
    #[serde(rename = "mjpeg")]
    Mjpeg,
}

impl StoredFormat {
    pub const fn pixel(self) -> PixelFormat {
        match self {
            Self::Gray8 => PixelFormat::Gray8,
            Self::Mjpeg => PixelFormat::Mjpeg,
        }
    }

    pub fn fourcc(self) -> FourCC {
        match self {
            Self::Gray8 => FourCC::new(b"GREY"),
            Self::Mjpeg => FourCC::new(b"MJPG"),
        }
    }

    pub const fn extension(self) -> &'static str {
        match self {
            Self::Gray8 => "pgm",
            Self::Mjpeg => "jpg",
        }
    }
}

impl TryFrom<PixelFormat> for StoredFormat {
    type Error = PixelFormat;

    fn try_from(value: PixelFormat) -> Result<Self, PixelFormat> {
        match value {
            PixelFormat::Gray8 => Ok(Self::Gray8),
            PixelFormat::Mjpeg => Ok(Self::Mjpeg),
            PixelFormat::Rgb8 => Err(value),
        }
    }
}

impl From<StoredFormat> for PixelFormat {
    fn from(value: StoredFormat) -> Self {
        value.pixel()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v4l2::V4l2Options;

    #[test]
    fn test_stored_format_serde_names_match_pixel_format() {
        for (stored, pixel) in [
            (StoredFormat::Gray8, PixelFormat::Gray8),
            (StoredFormat::Mjpeg, PixelFormat::Mjpeg),
        ] {
            assert_eq!(
                serde_json::to_string(&stored).unwrap(),
                serde_json::to_string(&pixel).unwrap()
            );
        }

        let options: V4l2Options =
            toml::from_str("device = \"/dev/video2\"\nformat = \"gray\"\nsize = [640, 360]")
                .unwrap();
        assert_eq!(options.format, StoredFormat::Gray8);
    }

    #[test]
    fn test_stored_format_fourcc_and_extension() {
        assert_eq!(StoredFormat::Gray8.fourcc(), FourCC::new(b"GREY"));
        assert_eq!(StoredFormat::Mjpeg.fourcc(), FourCC::new(b"MJPG"));
        assert_eq!(StoredFormat::Gray8.extension(), "pgm");
        assert_eq!(StoredFormat::Mjpeg.extension(), "jpg");
        assert_eq!(
            StoredFormat::try_from(PixelFormat::Rgb8),
            Err(PixelFormat::Rgb8)
        );
    }
}
