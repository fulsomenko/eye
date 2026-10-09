use std::fs;
use std::path::Path;

use eye_core::log::field;

use crate::ProbeError;
use crate::camera::UsbIdentity;
use crate::camera::usb_desc::{self, ExtensionUnit};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SysfsNode {
    pub name: String,
    pub number: u32,
    pub card: String,
    pub usb: Option<UsbIdentity>,
    pub extension_units: Vec<ExtensionUnit>,
}

pub(crate) fn scan(sysfs_class: &Path) -> Result<Vec<SysfsNode>, ProbeError> {
    let mut nodes = Vec::new();
    let entries = fs::read_dir(sysfs_class).map_err(|source| ProbeError::Io {
        path: sysfs_class.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| ProbeError::Io {
            path: sysfs_class.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();

        let number = match name
            .strip_prefix("video")
            .and_then(|n| n.parse::<u32>().ok())
        {
            Some(n) => n,
            None => continue,
        };

        let card_path = path.join("name");
        let card = read_trimmed(&card_path)?.unwrap_or_default();

        let (usb, descriptors) = match usb_identity(&path)? {
            Some((identity, raw)) => (Some(identity), Some(raw)),
            None => (None, None),
        };

        let extension_units = match (&usb, &descriptors) {
            (Some(identity), Some(raw)) => match usb_desc::extension_units(raw) {
                Ok(units) => units
                    .into_iter()
                    .filter(|u| u.interface == identity.interface)
                    .collect(),
                Err(error) => {
                    tracing::warn!(node = %name, %error, "failed to parse USB descriptors");
                    Vec::new()
                }
            },
            _ => Vec::new(),
        };

        tracing::debug!(
            node = %name,
            number,
            card = %card,
            usb = usb.is_some(),
            extension_units = extension_units.len(),
            "sysfs node scanned"
        );

        nodes.push(SysfsNode {
            name,
            number,
            card,
            usb,
            extension_units,
        });
    }
    nodes.sort_by_key(|n| n.number);
    Ok(nodes)
}

pub(crate) fn usb_identity(
    class_entry: &Path,
) -> Result<Option<(UsbIdentity, Vec<u8>)>, ProbeError> {
    let device_link = class_entry.join("device");
    let interface_dir = match fs::canonicalize(&device_link) {
        Ok(path) => path,
        Err(_) => {
            tracing::trace!(
                node = %class_entry.display(),
                { field::REASON } = "no_device_link",
                "node has no usb identity"
            );
            return Ok(None);
        }
    };

    let interface = match read_trimmed(&interface_dir.join("bInterfaceNumber"))? {
        Some(text) => match u8::from_str_radix(text.trim(), 16) {
            Ok(n) => n,
            Err(_) => {
                tracing::trace!(
                    node = %class_entry.display(),
                    { field::REASON } = "bad_interface_number",
                    text = %text,
                    "node has no usb identity"
                );
                return Ok(None);
            }
        },
        None => {
            tracing::trace!(
                node = %class_entry.display(),
                { field::REASON } = "no_interface_number",
                "node has no usb identity"
            );
            return Ok(None);
        }
    };

    let sysfs_device = match interface_dir.parent() {
        Some(parent) => parent.to_path_buf(),
        None => {
            tracing::trace!(
                node = %class_entry.display(),
                { field::REASON } = "no_parent_dir",
                "node has no usb identity"
            );
            return Ok(None);
        }
    };

    let vendor_id = parse_hex_u16(&sysfs_device.join("idVendor"))?;
    let product_id = parse_hex_u16(&sysfs_device.join("idProduct"))?;
    let descriptors =
        fs::read(sysfs_device.join("descriptors")).map_err(|source| ProbeError::Io {
            path: sysfs_device.join("descriptors"),
            source,
        })?;

    Ok(Some((
        UsbIdentity {
            vendor_id,
            product_id,
            interface,
            sysfs_device,
        },
        descriptors,
    )))
}

fn parse_hex_u16(path: &Path) -> Result<u16, ProbeError> {
    let text = fs::read_to_string(path).map_err(|source| ProbeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    u16::from_str_radix(text.trim(), 16).map_err(|_| ProbeError::Sysfs {
        path: path.to_path_buf(),
        reason: "not a hexadecimal u16",
    })
}

fn read_trimmed(path: &Path) -> Result<Option<String>, ProbeError> {
    match fs::read_to_string(path) {
        Ok(text) => Ok(Some(text.trim().to_string())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(ProbeError::Io {
            path: path.to_path_buf(),
            source,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    struct Fixture {
        root: std::path::PathBuf,
    }

    impl Fixture {
        fn build() -> Self {
            let root = std::env::temp_dir().join(format!(
                "eye-sysfs-{}-{}",
                std::process::id(),
                fixture_counter()
            ));
            let class_dir = root.join("class/video4linux");
            fs::create_dir_all(&class_dir).unwrap();

            let devices = root.join("devices/usb3/3-6");
            fs::create_dir_all(&devices).unwrap();
            fs::write(devices.join("idVendor"), "0c45\n").unwrap();
            fs::write(devices.join("idProduct"), "672c\n").unwrap();
            fs::write(devices.join("descriptors"), FIXTURE).unwrap();

            let if0 = devices.join("3-6:1.0");
            fs::create_dir_all(&if0).unwrap();
            fs::write(if0.join("bInterfaceNumber"), "00\n").unwrap();

            let if2 = devices.join("3-6:1.2");
            fs::create_dir_all(&if2).unwrap();
            fs::write(if2.join("bInterfaceNumber"), "02\n").unwrap();

            for (name, index, iface_dir) in [
                ("video0", 0, &if0),
                ("video1", 1, &if0),
                ("video2", 0, &if2),
                ("video3", 1, &if2),
            ] {
                let node_dir = class_dir.join(name);
                fs::create_dir_all(&node_dir).unwrap();
                fs::write(node_dir.join("name"), "Integrated_Webcam_HD: Integrate\n").unwrap();
                fs::write(node_dir.join("index"), format!("{index}\n")).unwrap();
                symlink(iface_dir, node_dir.join("device")).unwrap();
            }

            Self { root }
        }

        fn class_dir(&self) -> std::path::PathBuf {
            self.root.join("class/video4linux")
        }

        fn add_non_usb_node(&self, name: &str) {
            let node_dir = self.class_dir().join(name);
            fs::create_dir_all(&node_dir).unwrap();
            fs::write(node_dir.join("name"), "Non USB\n").unwrap();
            fs::write(node_dir.join("index"), "0\n").unwrap();

            let not_usb = self.root.join("devices/not-usb");
            fs::create_dir_all(&not_usb).unwrap();
            symlink(&not_usb, node_dir.join("device")).unwrap();
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn fixture_counter() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        COUNTER.fetch_add(1, Ordering::Relaxed)
    }

    const FIXTURE: &[u8] = include_bytes!("../../tests/fixtures/usb-0c45-672c.descriptors");

    #[test]
    fn test_sysfs_scan_maps_nodes_to_interfaces() {
        let fixture = Fixture::build();
        let nodes = scan(&fixture.class_dir()).expect("scan succeeds");

        let names: Vec<&str> = nodes.iter().map(|n| n.name.as_str()).collect();
        assert_eq!(names, vec!["video0", "video1", "video2", "video3"]);

        let video2 = nodes
            .iter()
            .find(|n| n.name == "video2")
            .expect("video2 present");
        let usb = video2.usb.as_ref().expect("video2 has usb identity");
        assert_eq!(usb.vendor_id, 0x0c45);
        assert_eq!(usb.product_id, 0x672c);
        assert_eq!(usb.interface, 2);
        assert_eq!(
            usb.sysfs_device,
            fs::canonicalize(fixture.root.join("devices/usb3/3-6")).unwrap()
        );

        assert_eq!(video2.extension_units.len(), 2);
        assert!(video2.extension_units.iter().all(|u| u.interface == 2));
    }

    #[test]
    fn test_metadata_node_pairs_by_interface() {
        use crate::camera::metadata_node_for;

        let fixture = Fixture::build();
        let nodes = scan(&fixture.class_dir()).expect("scan succeeds");
        let by_name = |name: &str| nodes.iter().find(|n| n.name == name).unwrap();
        let video0 = by_name("video0");
        let video1 = by_name("video1");
        let video2 = by_name("video2");
        let video3 = by_name("video3");
        let metas = [video1, video3];

        assert_eq!(
            metadata_node_for(video2, &metas, Path::new("/dev")),
            Some(std::path::PathBuf::from("/dev/video3"))
        );
        assert_eq!(
            metadata_node_for(video0, &metas, Path::new("/dev")),
            Some(std::path::PathBuf::from("/dev/video1"))
        );

        let no_usb = SysfsNode {
            name: "video9".to_string(),
            number: 9,
            card: String::new(),
            usb: None,
            extension_units: vec![],
        };
        assert_eq!(metadata_node_for(&no_usb, &metas, Path::new("/dev")), None);
    }

    #[test]
    fn test_non_usb_node_has_no_usb_identity() {
        let fixture = Fixture::build();
        fixture.add_non_usb_node("video9");
        let nodes = scan(&fixture.class_dir()).expect("scan succeeds");
        let video9 = nodes
            .iter()
            .find(|n| n.name == "video9")
            .expect("video9 present");
        assert_eq!(video9.usb, None);
        assert_eq!(video9.extension_units, vec![]);
    }

    #[test]
    fn test_logs_sysfs_node_scanned_at_debug() {
        use eye_log::Value;

        let fixture = Fixture::build();
        fixture.add_non_usb_node("video9");
        let video8_dir = fixture.class_dir().join("video8");
        fs::create_dir_all(&video8_dir).unwrap();
        fs::write(video8_dir.join("name"), "No link\n").unwrap();
        fs::write(video8_dir.join("index"), "0\n").unwrap();

        let (nodes, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            scan(&fixture.class_dir()).expect("scan succeeds")
        });
        assert_eq!(nodes.len(), 6);

        let scanned: Vec<_> = records
            .iter()
            .filter(|r| r.message == "sysfs node scanned")
            .collect();
        assert_eq!(scanned.len(), 6);
        for r in &scanned {
            assert_eq!(r.level, eye_log::Level::Debug);
        }

        let video2 = scanned
            .iter()
            .find(|r| r.fields.get("node") == Some(&Value::Str("video2".to_string())))
            .expect("video2 scanned");
        assert_eq!(video2.fields.get("usb"), Some(&Value::Bool(true)));
        assert_eq!(video2.fields.get("extension_units"), Some(&Value::U64(2)));

        for name in ["video8", "video9"] {
            let r = scanned
                .iter()
                .find(|r| r.fields.get("node") == Some(&Value::Str(name.to_string())))
                .unwrap_or_else(|| panic!("{name} scanned"));
            assert_eq!(r.fields.get("usb"), Some(&Value::Bool(false)));
        }

        let no_identity: Vec<_> = records
            .iter()
            .filter(|r| r.message == "node has no usb identity")
            .collect();
        assert_eq!(no_identity.len(), 2);
        for r in &no_identity {
            assert_eq!(r.level, eye_log::Level::Trace);
        }

        let video9_trace = no_identity
            .iter()
            .find(|r| {
                r.fields
                    .get("node")
                    .map(|v| matches!(v, Value::Str(s) if s.ends_with("video9")))
                    .unwrap_or(false)
            })
            .expect("video9 no-identity trace");
        assert_eq!(
            video9_trace.fields.get("reason"),
            Some(&Value::Str("no_interface_number".to_string()))
        );

        let video8_trace = no_identity
            .iter()
            .find(|r| {
                r.fields
                    .get("node")
                    .map(|v| matches!(v, Value::Str(s) if s.ends_with("video8")))
                    .unwrap_or(false)
            })
            .expect("video8 no-identity trace");
        assert_eq!(
            video8_trace.fields.get("reason"),
            Some(&Value::Str("no_device_link".to_string()))
        );
    }

    #[test]
    fn test_parse_hex_u16_reports_sysfs_path_on_garbage() {
        let fixture = Fixture::build();
        let path = fixture.root.join("idVendor");
        fs::write(&path, "zz\n").unwrap();

        let err = parse_hex_u16(&path).expect_err("garbage hex fails to parse");
        assert!(
            matches!(&err, ProbeError::Sysfs { path: p, reason: "not a hexadecimal u16" } if p.ends_with("idVendor"))
        );
        let message = err.to_string();
        assert!(message.starts_with("sysfs "));
        assert!(!message.contains("descriptor"));
    }
}
