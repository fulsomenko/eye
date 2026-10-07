{
  pkgs ? import <nixpkgs> {},
  rustToolchain ? pkgs.rustc,
}:

let
  # Loaded at runtime via dlopen by the Wayland client stack.
  runtimeLibs = with pkgs; lib.optionals stdenv.hostPlatform.isLinux [
    wayland
    libxkbcommon
  ];
in

pkgs.mkShell {
  name = "eye-rust-shell";

  nativeBuildInputs = with pkgs; [
    pkg-config
    rustPlatform.bindgenHook
  ];

  buildInputs = with pkgs; [
    rustToolchain
    cargo-watch
    cargo-edit
    cargo-audit
    cargo-nextest
    cargo-llvm-cov
    bacon
    jq
  ] ++ lib.optionals stdenv.hostPlatform.isLinux [
    v4l-utils
    ffmpeg-headless
    wayland
    wayland-protocols
    wayland-scanner
    libxkbcommon
    onnxruntime
  ] ++ runtimeLibs;

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;
  ORT_DYLIB_PATH = "${pkgs.onnxruntime}/lib/libonnxruntime.so";

  shellHook = ''
    export RUST_BACKTRACE=1
    echo "Eye Development Environment"
    echo "📦 Cargo: $(cargo --version)"
    echo "🦀 Rustc: $(rustc --version)"
  '';
}
