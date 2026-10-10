pub fn install() -> anyhow::Result<crossbeam_channel::Receiver<()>> {
    use signal_hook::consts::signal::{SIGINT, SIGTERM};
    use signal_hook::iterator::Signals;

    let mut signals = Signals::new([SIGINT, SIGTERM])?;
    let (tx, rx) = crossbeam_channel::bounded(1);
    eye_core::log::spawn_in_current_span("eye-signals", move || {
        let mut requested = false;
        for signal in signals.forever() {
            if requested {
                std::process::exit(130);
            }
            requested = true;
            tracing::info!(signal, "shutting down; send the signal again to force");
            let _ = tx.try_send(());
        }
    })?;
    Ok(rx)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use signal_hook::consts::signal::SIGTERM;

    use super::*;

    #[test]
    fn test_sigterm_delivers_shutdown() {
        let rx = install().unwrap();
        signal_hook::low_level::raise(SIGTERM).unwrap();
        assert_eq!(rx.recv_timeout(Duration::from_secs(2)), Ok(()));
    }
}
