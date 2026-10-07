# eye

Webcam gaze tracking for Linux laptops. A pluggable pipeline turns frames from the built-in RGB and IR cameras into the point on the screen you are looking at, and a Wayland layer-shell overlay draws it on the desktop. Every stage (capture, detection, estimation, calibration, filtering) has swappable implementations chosen from config, and a benchmark harness scores them against recorded ground truth.

## Status

Pre-alpha. The pipeline is under construction.

## Development

```sh
direnv allow        # or: nix develop
cargo nextest run --workspace
nix build
```

## Hardware

Developed on a Dell Latitude 7420 with an IR (Windows Hello) camera, under Hyprland. Other laptops and compositors are untested.

## Privacy

`recordings/` holds face images captured for calibration and benchmarking. It is git-ignored and must never be shared.

## License

Licensed under Apache-2.0, see [LICENSE](LICENSE).
