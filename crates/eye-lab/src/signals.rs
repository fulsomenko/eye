use std::sync::{Arc, atomic::AtomicBool};

pub fn install(cancel: &Arc<AtomicBool>) -> std::io::Result<()> {
    use signal_hook::consts::signal::{SIGINT, SIGTERM};
    for signal in [SIGINT, SIGTERM] {
        signal_hook::flag::register_conditional_shutdown(signal, 130, Arc::clone(cancel))?;
        signal_hook::flag::register(signal, Arc::clone(cancel))?;
    }
    Ok(())
}
