{
  lib,
  pkgs,
  rustPlatform,
  makeWrapper,
  onnxruntime,
  src,
  gitRev ? null,
}: let
  cargoToml = lib.importTOML ./Cargo.toml;
in
  rustPlatform.buildRustPackage {
    pname = "eye";
    inherit (cargoToml.workspace.package) version;
    inherit src;

    cargoLock.lockFile = ./Cargo.lock;

    nativeBuildInputs = [pkgs.pkg-config rustPlatform.bindgenHook makeWrapper];

    cargoBuildFlags = ["--package" "eye-app"];
    doCheck = false;

    preBuild = lib.optionalString (gitRev != null) ''
      export EYE_GIT_REV="${gitRev}"
    '';

    postInstall = ''
      wrapProgram $out/bin/eye --set-default ORT_DYLIB_PATH ${onnxruntime}/lib/libonnxruntime.so
    '';

    meta = {
      inherit (cargoToml.workspace.package) description;
      license = lib.licenses.asl20;
      mainProgram = "eye";
      platforms = lib.platforms.linux;
    };
  }
