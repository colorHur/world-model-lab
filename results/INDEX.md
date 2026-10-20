# results 索引（自动生成，勿手改）

由 `wmlab/eval/results_io.py::rebuild_index()` 生成。手改会被下一次写入覆盖。

| 状态 | 含义 |
|---|---|
| `current` | 该逻辑名的**最新**版本（引用数字请指这里） |
| `superseded` | 旧版本，**保留不删**，仅为可追溯 |
| `v1(frozen)` | P0-3 之前写在 `results/` 根目录的 v1 产物，**原地冻结**，未经 run-id 复跑 |

## 证据（`results/runs/<run_id>/` 与冻结的 v1）

| 逻辑名 | 版本 | 状态 | 文件时间 | 大小 | sha256 | 路径 |
|---|---|---|---|---|---|---|
| `42_t12_four_arms_summary` | v1 | **superseded** | 2026-10-19 11:28 | 90055 B | `2bbeb6622426` | `results/42_t12_four_arms_summary.json` |
| `42_t12_four_arms_summary` | 20261019-134106-ef67549 | **current** | 2026-10-19 13:41 | 90055 B | `5012d4be62f5` | `results/runs/20261019-134106-ef67549/42_t12_four_arms_summary.json` |
| `42_t12_four_arms_summary.v1_absorbed_meta_err` | v1 | **v1(frozen)** | 2026-10-19 11:27 | 89860 B | `2a07e64160fd` | `results/42_t12_four_arms_summary.v1_absorbed_meta_err.json` |
| `43_fix_abc_selfcheck` | v1 | **v1(frozen)** | 2026-10-19 11:29 | 720 B | `dc36443edd07` | `results/43_fix_abc_selfcheck.json` |
| `44_lookahead_selfcheck` | v1 | **v1(frozen)** | 2026-10-19 11:29 | 2154 B | `cbad0af99004` | `results/44_lookahead_selfcheck.json` |
| `45_t14_horizon_arms` | v1 | **v1(frozen)** | 2026-10-19 04:32 | 95366 B | `5bc20c38fc88` | `results/45_t14_horizon_arms.json` |
| `46_t14_failure_timescale` | v1 | **v1(frozen)** | 2026-10-19 04:32 | 5337 B | `9612beee95c7` | `results/46_t14_failure_timescale.json` |
| `47_t15_normalized_age` | v1 | **v1(frozen)** | 2026-10-19 08:52 | 84722 B | `cf1cf2b35af4` | `results/47_t15_normalized_age.json` |
| `48_t16_quota_pacing` | v1 | **v1(frozen)** | 2026-10-19 08:50 | 420197 B | `c36e2da6971a` | `results/48_t16_quota_pacing.json` |
| `50_p0_wiring_selfcheck` | v1 | **v1(frozen)** | 2026-10-19 11:28 | 520 B | `44cc9901b84a` | `results/50_p0_wiring_selfcheck.json` |
| `52_t21_paced_matching_ablation` | 20261019-154839-132f110 | **current** | 2026-10-19 15:48 | 167442 B | `99fe7398349a` | `results/runs/20261019-154839-132f110/52_t21_paced_matching_ablation.json` |
| `56_t23b_compensation_benefit` | 20261020-003551-daacb33 | **superseded** | 2026-10-20 00:35 | 322111 B | `b6fa72a84531` | `results/runs/20261020-003551-daacb33/56_t23b_compensation_benefit.json` |
| `56_t23b_compensation_benefit` | 20261020-003811-daacb33 | **superseded** | 2026-10-20 00:38 | 339134 B | `891de28d1ed9` | `results/runs/20261020-003811-daacb33/56_t23b_compensation_benefit.json` |
| `56_t23b_compensation_benefit` | 20261020-004151-daacb33 | **current** | 2026-10-20 00:41 | 341228 B | `1d1d9f6278f9` | `results/runs/20261020-004151-daacb33/56_t23b_compensation_benefit.json` |
| `57_t24_shared_ab` | 20261020-025036-62663b3 | **current** | 2026-10-20 02:50 | 339059 B | `6a86ca85bb6e` | `results/runs/20261020-025036-62663b3/57_t24_shared_ab.json` |
| `58_p3_confirmation` | 20261020-050355-613f4be | **current** | 2026-10-20 05:03 | 625811 B | `504d8235fb97` | `results/runs/20261020-050355-613f4be/58_p3_confirmation.json` |
| `59_p3_boundary` | 20261020-071627-58ee7b7 | **superseded** | 2026-10-20 07:16 | 398607 B | `dccf9b811f4b` | `results/runs/20261020-071627-58ee7b7/59_p3_boundary.json` |
| `59_p3_boundary` | 20261020-072122-58ee7b7 | **current** | 2026-10-20 07:21 | 398626 B | `b6617d27e780` | `results/runs/20261020-072122-58ee7b7/59_p3_boundary.json` |
| `59b_p3_boundary_refine` | 20261020-071807-58ee7b7 | **superseded** | 2026-10-20 07:18 | 78080 B | `a050a5a767de` | `results/runs/20261020-071807-58ee7b7/59b_p3_boundary_refine.json` |
| `59b_p3_boundary_refine` | 20261020-071854-58ee7b7 | **current** | 2026-10-20 07:18 | 78061 B | `925230560bdf` | `results/runs/20261020-071854-58ee7b7/59b_p3_boundary_refine.json` |
| `60_p3_window_shift` | 20261020-094752-7fcaaa5 | **current** | 2026-10-20 09:47 | 594015 B | `34a754d78498` | `results/runs/20261020-094752-7fcaaa5/60_p3_window_shift.json` |
| `66_identifiability` | 20261020-183802-1c59479 | **current** | 2026-10-20 18:38 | 28016 B | `636dd24cbf52` | `results/runs/20261020-183802-1c59479/66_identifiability.json` |
| `67_cf_action` | 20261020-204741-2fbe298 | **superseded** | 2026-10-20 20:47 | 5562 B | `69bf81a854cb` | `results/runs/20261020-204741-2fbe298/67_cf_action.json` |
| `67_cf_action` | 20261020-204836-2fbe298 | **current** | 2026-10-20 20:48 | 12384 B | `024ff94226a0` | `results/runs/20261020-204836-2fbe298/67_cf_action.json` |
| `68_t27_gated_fallback` | 20261020-225901-ece860c | **current** | 2026-10-20 22:59 | 141889 B | `67f0721eb1dc` | `results/runs/20261020-225901-ece860c/68_t27_gated_fallback.json` |
| `69_t28_confirm` | 20261021-010622-b7d0421 | **superseded** | 2026-10-21 01:06 | 64043 B | `f0379eb5b784` | `results/runs/20261021-010622-b7d0421/69_t28_confirm.json` |
| `69_t28_confirm` | 20261021-010749-b7d0421 | **current** | 2026-10-21 01:07 | 64040 B | `6566d67ba983` | `results/runs/20261021-010749-b7d0421/69_t28_confirm.json` |

## 自检快照（`results/selfcheck/`，判定不变则不重写）

| 快照 | 文件时间 | 大小 | sha256 |
|---|---|---|---|
| `43_fix_abc_selfcheck` | 2026-10-19 13:37 | 720 B | `dc36443edd07` |
| `44_lookahead_selfcheck` | 2026-10-19 13:38 | 2154 B | `cbad0af99004` |
| `50_p0_wiring_selfcheck` | 2026-10-19 13:38 | 520 B | `44cc9901b84a` |
| `51_p0_results_io_selfcheck` | 2026-10-19 13:37 | 4260 B | `f94f5b0e510e` |
