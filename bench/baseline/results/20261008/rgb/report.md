# eye bench report

eye 0.1.0. Grids 3x3, 4x4; boundary margin 20 px; dropout bin 200 ms.
`proc ms` is pipeline processing time per FrameSet, not end-to-end latency (see `eye run --stats`).

| pipeline | calib | session | status | samples | err mean deg | err p95 deg | acc deg | prec deg | err mean px | 3x3 hit | 4x4 hit | proc p50 ms | proc p95 ms | dropout |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rgb | none | 20261008T201638Z | ok | 384 | 20.60 | 27.04 | 20.59 | 0.31 | 987.7 | 0.0 % | 0.0 % | 39.1 | 63.2 | 0.0 % |
| rgb | none | 20261008T201709Z | ok | 384 | 21.84 | 27.81 | 21.82 | 0.52 | 1054.0 | 0.0 % | 0.0 % | 25.4 | 37.3 | 0.0 % |
| rgb | none | **all** | ok | 768 | 21.22 | 27.64 | 21.21 | 0.43 | 1020.8 | 0.0 % | 0.0 % | 35.4 | 55.5 | 0.0 % |
| rgb | loto | 20261008T201638Z | ok | 384 | 3.58 | 7.80 | 3.50 | 0.46 | 179.0 | 75.0 % | 56.2 % | 39.1 | 63.2 | 0.0 % |
| rgb | loto | 20261008T201709Z | ok | 384 | 4.34 | 8.92 | 4.19 | 0.94 | 218.6 | 62.5 % | 43.8 % | 25.4 | 37.3 | 0.0 % |
| rgb | loto | **all** | ok | 768 | 3.96 | 8.39 | 3.85 | 0.74 | 198.8 | 68.8 % | 50.0 % | 35.4 | 55.5 | 0.0 % |

## Accuracy targets (aggregate rows)

| pipeline | calib | 3x3 hit >= 90 % | 4x4 hit >= 90 % | err mean deg | err p95 deg | mean < 2 deg (stretch) |
|---|---|---|---|---:|---:|---|
| rgb | none | no | no | 21.22 | 27.64 | no |
| rgb | loto | no | no | 3.96 | 8.39 | no |
