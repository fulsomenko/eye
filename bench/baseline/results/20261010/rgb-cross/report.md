# eye bench report

eye 0.1.0. Grids 3x3, 4x4; boundary margin 20 px; dropout bin 200 ms.
`proc ms` is pipeline processing time per FrameSet, not end-to-end latency (see `eye run --stats`).

| pipeline | calib | session | status | samples | err mean deg | err p95 deg | acc deg | prec deg | err mean px | 3x3 hit | 4x4 hit | proc p50 ms | proc p95 ms | dropout |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rgb | none | 20261008T201638Z | ok | 384 | 20.63 | 27.13 | 20.62 | 0.27 | 988.9 | 0.0 % | 0.0 % | 167.0 | 298.5 | 0.0 % |
| rgb | none | 20261008T201709Z | ok | 384 | 21.86 | 27.94 | 21.83 | 0.41 | 1054.7 | 0.0 % | 0.0 % | 155.5 | 448.7 | 0.0 % |
| rgb | none | **all** | ok | 768 | 21.25 | 27.77 | 21.22 | 0.35 | 1021.8 | 0.0 % | 0.0 % | 166.3 | 407.0 | 0.0 % |
| rgb | loto | 20261008T201638Z | ok | 384 | 2.64 | 5.31 | 2.58 | 0.33 | 133.0 | 75.0 % | 62.5 % | 167.0 | 298.5 | 0.0 % |
| rgb | loto | 20261008T201709Z | ok | 384 | 3.27 | 7.17 | 3.13 | 0.62 | 164.3 | 50.0 % | 37.5 % | 155.5 | 448.7 | 0.0 % |
| rgb | loto | **all** | ok | 768 | 2.95 | 5.72 | 2.85 | 0.50 | 148.7 | 62.5 % | 50.0 % | 166.3 | 407.0 | 0.0 % |
| rgb | cross | 20261008T201638Z | ok | 384 | 2.77 | 4.84 | 2.63 | 0.34 | 139.1 | 81.2 % | 68.8 % | 167.0 | 298.5 | 0.0 % |
| rgb | cross | 20261008T201709Z | ok | 384 | 3.05 | 6.15 | 2.91 | 0.64 | 152.8 | 68.8 % | 68.8 % | 155.5 | 448.7 | 0.0 % |
| rgb | cross | **all** | ok | 768 | 2.91 | 5.44 | 2.77 | 0.51 | 145.9 | 75.0 % | 68.8 % | 166.3 | 407.0 | 0.0 % |

## Accuracy targets (aggregate rows)

| pipeline | calib | 3x3 hit >= 90 % | 4x4 hit >= 90 % | err mean deg | err p95 deg | mean < 2 deg (stretch) |
|---|---|---|---|---:|---:|---|
| rgb | none | no | no | 21.25 | 27.77 | no |
| rgb | loto | no | no | 2.95 | 5.72 | no |
| rgb | cross | no | no | 2.91 | 5.44 | no |
