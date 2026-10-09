# eye

Webcam gaze tracking. A pluggable pipeline turns frames from the built-in RGB and IR cameras into the point on the screen you are looking at, and a Wayland layer-shell overlay draws it on the desktop. Every stage (capture, detection, estimation, calibration, filtering) has swappable implementations chosen from config, and a benchmark harness scores them against recorded ground truth.

## Status

Pre-alpha. The pipeline is under construction.

## Development

```sh
cargo build
cargo test
```

## Calibration

`eye calibrate` shows a dot on each target in turn. A thin arc sweeps clockwise around the dot as it counts down the time the eye needs to settle on it; the dot then fades from white to green while samples are being taken, and holds solid green until the next dot appears.

## Logs

`--log-file auto` writes JSON Lines to `$XDG_STATE_HOME/eye/logs/<run.id>.jsonl` (default `~/.local/state/eye/logs/`). `--log-level`/`EYE_LOG` set the terminal filter. Log files are local artifacts like `recordings/`: nothing is uploaded.

## Hardware

Developed on a Dell Latitude 7420 with an IR (Windows Hello) camera, under Hyprland. Other laptops and compositors are untested.

## License

Licensed under the Apache License, Version 2.0, see [LICENSE](LICENSE).
