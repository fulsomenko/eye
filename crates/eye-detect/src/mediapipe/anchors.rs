use crate::mediapipe::blazeface::{DETECTOR_INPUT, NUM_ANCHORS};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Anchor {
    pub x: f32,
    pub y: f32,
}

pub fn short_range_anchors() -> Vec<Anchor> {
    let mut anchors = Vec::with_capacity(NUM_ANCHORS);
    for (stride, per_cell) in [(8usize, 2usize), (16, 6)] {
        let n = DETECTOR_INPUT / stride;
        for y in 0..n {
            for x in 0..n {
                let a = Anchor {
                    x: (x as f32 + 0.5) / n as f32,
                    y: (y as f32 + 0.5) / n as f32,
                };
                anchors.extend(std::iter::repeat_n(a, per_cell));
            }
        }
    }
    anchors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_range_anchors_count_is_896() {
        assert_eq!(short_range_anchors().len(), NUM_ANCHORS);
    }

    #[test]
    fn test_anchor_layout_matches_mediapipe() {
        let anchors = short_range_anchors();
        assert_eq!(
            anchors[0],
            Anchor {
                x: 0.03125,
                y: 0.03125
            }
        );
        assert_eq!(
            anchors[1],
            Anchor {
                x: 0.03125,
                y: 0.03125
            }
        );
        assert_eq!(
            anchors[512],
            Anchor {
                x: 0.0625,
                y: 0.0625
            }
        );
        assert_eq!(
            anchors[895],
            Anchor {
                x: 0.9375,
                y: 0.9375
            }
        );
    }
}
