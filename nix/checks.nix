{
  pkgs,
  rustToolchain,
  eye,
  src,
}: let
  cargoCheck = name: extraInputs: command:
    eye.overrideAttrs (old: {
      pname = "eye-${name}";
      nativeBuildInputs = old.nativeBuildInputs ++ extraInputs;
      buildPhase = ''
        runHook preBuild
        ${command}
        runHook postBuild
      '';
      doCheck = false;
      installPhase = "touch $out";
    });
in {
  package = eye;

  clippy = cargoCheck "clippy" [] ''
    cargo clippy --offline --workspace --all-targets --all-features -- -D warnings
  '';

  nextest = cargoCheck "nextest" [pkgs.cargo-nextest] ''
    cargo nextest run --offline --workspace --all-features --no-tests=warn
  '';

  fmt = pkgs.runCommand "eye-fmt" {nativeBuildInputs = [rustToolchain];} ''
    cd ${src}
    HOME=$TMPDIR cargo fmt --all -- --check
    touch $out
  '';

  crate-layers = pkgs.runCommand "eye-crate-layers" {nativeBuildInputs = [rustToolchain pkgs.jq pkgs.bash];} ''
    cd ${src}
    HOME=$TMPDIR bash scripts/check-crate-layers.sh
    touch $out
  '';

  no-captures = pkgs.runCommand "eye-no-captures" {nativeBuildInputs = [pkgs.bash pkgs.findutils];} ''
    cd ${src}
    bash scripts/check-no-captures.sh
    touch $out
  '';
}
