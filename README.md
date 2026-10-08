# eye

Webcam gaze tracking. A pluggable pipeline turns frames from the built-in RGB and IR cameras into the point on the screen you are looking at, and a Wayland layer-shell overlay draws it on the desktop. Every stage (capture, detection, estimation, calibration, filtering) has swappable implementations chosen from config, and a benchmark harness scores them against recorded ground truth.

## Status

Pre-alpha. The pipeline is under construction.

## Development

```sh
cargo build
cargo test
```

## Hardware

Developed on a Dell Latitude 7420 with an IR (Windows Hello) camera, under Hyprland. Other laptops and compositors are untested.

## License

Licensed under the Apache License, Version 2.0, see [LICENSE](LICENSE).
