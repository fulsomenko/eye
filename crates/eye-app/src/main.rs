#![forbid(unsafe_code)]

fn main() {
    println!("eye {}", env!("CARGO_PKG_VERSION"));
}
