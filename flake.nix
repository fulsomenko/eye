{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    rust-overlay,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachSystem ["x86_64-linux" "aarch64-linux"] (
      system: let
        overlays = [(import rust-overlay)];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = ["rust-src" "rust-analyzer" "clippy" "rustfmt" "llvm-tools-preview"];
        };

        rustPlatform = pkgs.makeRustPlatform {
          cargo = rustToolchain;
          rustc = rustToolchain;
        };

        eye = pkgs.callPackage ./default.nix {
          inherit rustPlatform;
          src = self;
          gitRev = self.rev or null;
        };

        models = pkgs.callPackage ./nix/models.nix {};
      in {
        devShells.default = import ./shell.nix {
          inherit pkgs rustToolchain models;
        };

        packages = {
          default = eye;
          inherit eye;
          mediapipe-models = models;
        };

        checks = import ./nix/checks.nix {
          inherit pkgs rustToolchain eye;
          src = self;
        };

        formatter = pkgs.alejandra;
      }
    );
}
