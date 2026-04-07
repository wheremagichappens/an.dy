from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CampaignConfig:
    user_id_col: str = "user_id"
    treatment_col: str = "treated"
    date_col: str = "week_start"
    outcome_col: str = "weekly_trips"
    post_col: str = "post"
    pre_period_end: str = "2025-09-01"
    post_period_start: str = "2025-09-08"
    caliper: float = 0.05
    random_state: int = 42
    output_dir: Path = Path("artifacts")
