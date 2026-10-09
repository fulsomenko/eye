use crate::ProbeError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Guid(pub [u8; 16]);

impl Guid {
    pub const fn from_fields(d1: u32, d2: u16, d3: u16, d4: [u8; 8]) -> Self {
        Self([
            d1 as u8,
            (d1 >> 8) as u8,
            (d1 >> 16) as u8,
            (d1 >> 24) as u8,
            d2 as u8,
            (d2 >> 8) as u8,
            d3 as u8,
            (d3 >> 8) as u8,
            d4[0],
            d4[1],
            d4[2],
            d4[3],
            d4[4],
            d4[5],
            d4[6],
            d4[7],
        ])
    }
}

impl std::fmt::Display for Guid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let b = &self.0;
        write!(
            f,
            "{{{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}}}",
            b[3],
            b[2],
            b[1],
            b[0],
            b[5],
            b[4],
            b[7],
            b[6],
            b[8],
            b[9],
            b[10],
            b[11],
            b[12],
            b[13],
            b[14],
            b[15],
        )
    }
}

impl serde::Serialize for Guid {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ExtensionUnit {
    pub interface: u8,
    pub unit_id: u8,
    pub guid: Guid,
    pub num_controls: u8,
    /// Selectors advertised in bmControls: bit n set => selector n + 1.
    pub selectors: Vec<u8>,
}

pub fn extension_units(d: &[u8]) -> Result<Vec<ExtensionUnit>, ProbeError> {
    let err = |offset, reason| ProbeError::Descriptor { offset, reason };
    let mut units = Vec::new();
    let mut vc_interface: Option<u8> = None;
    let mut i = 0;
    while i < d.len() {
        let len = d[i] as usize;
        if len < 2 || i + len > d.len() {
            return Err(err(i, "bLength out of range"));
        }
        let desc = &d[i..i + len];
        match desc[1] {
            0x04 if len >= 9 => {
                vc_interface = (desc[5] == 0x0e && desc[6] == 0x01).then_some(desc[2]);
            }
            0x24 if len >= 3 && desc[2] == 0x06 => {
                if let Some(interface) = vc_interface {
                    let unit =
                        parse_xu(interface, desc).ok_or_else(|| err(i, "short extension unit"))?;
                    tracing::trace!(
                        interface = unit.interface,
                        unit_id = unit.unit_id,
                        guid = %unit.guid,
                        num_controls = unit.num_controls,
                        selectors = unit.selectors.len(),
                        "extension unit parsed"
                    );
                    units.push(unit);
                }
            }
            _ => {}
        }
        i += len;
    }
    Ok(units)
}

fn parse_xu(interface: u8, desc: &[u8]) -> Option<ExtensionUnit> {
    let p = *desc.get(21)? as usize;
    let n = *desc.get(22 + p)? as usize;
    let bm = desc.get(23 + p..23 + p + n)?;
    let selectors = (0..n * 8)
        .filter(|bit| bm[bit / 8] & (1 << (bit % 8)) != 0)
        .map(|bit| bit as u8 + 1)
        .collect();
    Some(ExtensionUnit {
        interface,
        unit_id: desc[3],
        guid: Guid(desc[4..20].try_into().ok()?),
        num_controls: desc[20],
        selectors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &[u8] = include_bytes!("../../tests/fixtures/usb-0c45-672c.descriptors");

    #[test]
    fn test_descriptor_fixture_bytes_are_the_real_ones() {
        assert_eq!(FIXTURE.len(), 1065);
        assert_eq!(
            &FIXTURE[0x32f..0x32f + 27],
            &[
                0x1b, 0x24, 0x06, 0x04, 0xdc, 0x95, 0x3f, 0x0f, 0x32, 0x26, 0x4e, 0x4c, 0x92, 0xc9,
                0xa0, 0x47, 0x82, 0xf4, 0x3b, 0xc8, 0x10, 0x01, 0x03, 0x02, 0x20, 0x01, 0x00,
            ]
        );
    }

    #[test]
    fn test_fixture_yields_five_extension_units() {
        let units = extension_units(FIXTURE).expect("fixture parses");
        let got: Vec<(u8, u8, String, u8, Vec<u8>)> = units
            .iter()
            .map(|u| {
                (
                    u.interface,
                    u.unit_id,
                    u.guid.to_string(),
                    u.num_controls,
                    u.selectors.clone(),
                )
            })
            .collect();
        assert_eq!(
            got,
            vec![
                (
                    0,
                    3,
                    "{28f03370-6311-4a2e-ba2c-6890eb334016}".to_string(),
                    8,
                    vec![1, 2, 3, 4, 5, 6, 8]
                ),
                (
                    0,
                    4,
                    "{0fb885c3-68c2-4547-90f7-8f47579d95fc}".to_string(),
                    8,
                    vec![1, 2, 3, 4]
                ),
                (
                    0,
                    5,
                    "{0f3f95dc-2632-4c4e-92c9-a04782f43bc8}".to_string(),
                    16,
                    vec![]
                ),
                (
                    2,
                    3,
                    "{28f03370-6311-4a2e-ba2c-6890eb334016}".to_string(),
                    8,
                    vec![]
                ),
                (
                    2,
                    4,
                    "{0f3f95dc-2632-4c4e-92c9-a04782f43bc8}".to_string(),
                    16,
                    vec![6, 9]
                ),
            ]
        );
    }

    #[test]
    fn test_guid_display_uses_mixed_endian_fields() {
        let bytes: [u8; 16] = [
            0xdc, 0x95, 0x3f, 0x0f, 0x32, 0x26, 0x4e, 0x4c, 0x92, 0xc9, 0xa0, 0x47, 0x82, 0xf4,
            0x3b, 0xc8,
        ];
        let guid = Guid(bytes);
        assert_eq!(guid.to_string(), "{0f3f95dc-2632-4c4e-92c9-a04782f43bc8}");
        assert_eq!(
            guid,
            Guid::from_fields(
                0x0f3f95dc,
                0x2632,
                0x4c4e,
                [0x92, 0xc9, 0xa0, 0x47, 0x82, 0xf4, 0x3b, 0xc8]
            )
        );
    }

    #[test]
    fn test_zero_length_descriptor_is_error() {
        let result = extension_units(&[0x00, 0x24]);
        assert!(matches!(
            result,
            Err(ProbeError::Descriptor { offset: 0, .. })
        ));
    }

    #[test]
    fn test_overrunning_descriptor_is_error() {
        let truncated = &FIXTURE[..0x330];
        let result = extension_units(truncated);
        assert!(matches!(
            result,
            Err(ProbeError::Descriptor { offset: 0x32f, .. })
        ));
    }

    #[test]
    fn test_xu_outside_videocontrol_is_ignored() {
        let mut buf = vec![9u8, 0x04, 0x00, 0x00, 0x00, 0x0e, 0x02, 0x00, 0x00];
        buf.extend_from_slice(&[
            0x1b, 0x24, 0x06, 0x04, 0xdc, 0x95, 0x3f, 0x0f, 0x32, 0x26, 0x4e, 0x4c, 0x92, 0xc9,
            0xa0, 0x47, 0x82, 0xf4, 0x3b, 0xc8, 0x10, 0x01, 0x03, 0x02, 0x20, 0x01, 0x00,
        ]);
        let result = extension_units(&buf);
        assert_eq!(result.expect("parses"), vec![]);
    }

    #[test]
    fn test_logs_extension_unit_parsed_at_trace() {
        use eye_log::Value;

        let (units, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            extension_units(FIXTURE).expect("fixture parses")
        });
        assert_eq!(units.len(), 5);

        let parsed: Vec<_> = records
            .iter()
            .filter(|r| r.message == "extension unit parsed")
            .collect();
        assert_eq!(parsed.len(), 5);

        let target = parsed
            .iter()
            .find(|r| {
                r.fields.get("interface") == Some(&Value::U64(2))
                    && r.fields.get("unit_id") == Some(&Value::U64(4))
            })
            .expect("interface 2 unit 4 parsed");
        assert_eq!(target.level, eye_log::Level::Trace);
        assert_eq!(target.fields.get("selectors"), Some(&Value::U64(2)));
        assert_eq!(
            target.fields.get("guid"),
            Some(&Value::Str(
                "{0f3f95dc-2632-4c4e-92c9-a04782f43bc8}".to_string()
            ))
        );
    }
}
