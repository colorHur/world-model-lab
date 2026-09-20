"""评测指标与可靠视界。"""

from .aoi import (NmseCurve, expected_nmse_analytic, geometric_age_pmf,
                  geometric_age_stats, jensen_gap, loss_prob_for_mean_age,
                  loss_prob_for_tail_risk, nmse_at_mean_age, uniform_age_pmf,
                  uniform_age_stats)
from .metrics import (coverage, ece, mse, nmse, per_step_mse, per_step_nmse,
                      reliable_horizon)
from .tracking import (StateTracker, TrackingResult, lossy_schedule,
                       periodic_schedule, run_tracking)

__all__ = [
    # 基础指标
    "mse", "nmse", "reliable_horizon", "coverage", "ece",
    # ★ 逐点口径（H* 必须用这个；见 imagine.py 的口径说明）
    "per_step_mse", "per_step_nmse",
    # X24 跟踪
    "StateTracker", "TrackingResult", "run_tracking",
    "periodic_schedule", "lossy_schedule",
    # X26 AoI 解析
    "NmseCurve", "geometric_age_pmf", "uniform_age_pmf",
    "geometric_age_stats", "uniform_age_stats",
    "expected_nmse_analytic", "nmse_at_mean_age", "jensen_gap",
    "loss_prob_for_mean_age", "loss_prob_for_tail_risk",
]
