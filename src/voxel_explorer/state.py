"""Explorer runtime state (target tracking + drive bookkeeping)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from src.voxel_nav import Serial


@dataclass
class ExplorerState:
    target: Optional[Serial] = None
    target_source: Optional[Serial] = None     # the Reachable node we
                                               # discovered the target from
    e_count: float = 0.0
    last_distance: float = math.inf
    last_cell: Optional[Serial] = None         # for no-progress watchdog
    last_progress_t: float = 0.0
    action: str = "idle"
    # consecutive give-ups without changing voxel cell. when this gets high
    # we are stuck on a perch and the BFS keeps handing us unreachable
    # frontiers nearby, so targeting blacklists a radius around us to force
    # picking a frontier further away.
    consec_giveups_in_cell: int = 0
    giveup_cell: Optional[Serial] = None
