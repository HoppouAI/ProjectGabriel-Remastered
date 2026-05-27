"""Tunables for the raycast-driven wanderer."""

TARGET_FPS = 20  # raycasts come in at avatar tick rate, we can poll fast

DEFAULT_CFG = {
    # forward speed when path is clear
    "forward_speed": 0.6,
    # turn rates
    "turn_speed_avoid": 1.0,    # max turn rate when escaping a wall
    "turn_speed_steer": 0.6,    # max gradient-bias turn while moving
    "turn_speed_random": 0.6,   # exploration turns

    # raycast clearance thresholds (meters)
    "stop_distance": 0.8,       # below this on FwdNear -> reverse + commit turn
    "slow_distance": 2.0,       # start scaling speed down below this
    "cruise_distance": 3.5,     # full speed above this on Fwd
    "side_reference": 3.0,      # side distances are normalized against this

    # how long to stay in dedicated escape modes (seconds)
    "wall_commit_seconds": 1.4,
    "ledge_commit_seconds": 1.0,
    "uturn_commit_seconds": 2.5,    # full ~180 turn when dead-ended

    # dead-end detection (all forward-cone rays below this -> u-turn)
    "deadend_distance": 1.4,
    # escalation: repeated wall hits in this window promote to u-turn
    "escalation_window": 6.0,
    "escalation_threshold": 2,      # this many walls in window -> u-turn

    # exploration behavior
    "random_turn_chance": 0.02,
    "min_straight_time": 12.0,
    "max_straight_time": 25.0,
    "jump_chance": 0.012,

    # stuck detection
    "stuck_velocity_threshold": 0.05,
    "stuck_frames_to_reverse": 10,  # ~0.5s at 20fps
    "stuck_frames_to_jump": 40,     # ~2s at 20fps

    # smoothing
    "smoothing_alpha": 0.5,

    # auto-resume after silence
    "auto_resume_seconds": 30.0,

    # map-aware (curiosity) wandering. enabled automatically when the
    # mapping service is running and the graph has enough cells.
    "map_mode_min_cells": 30,            # need at least this many reachable cells
    "map_mode_min_radius": 3.0,          # candidate cells must be at least this far
    "map_mode_max_radius": 20.0,         # ...and no further than this
    "map_mode_sample_count": 80,         # how many candidates to score per pick
    "map_mode_softmax_temp": 0.6,        # higher = more random pick
    "map_mode_dwell_min_s": 0.8,         # pause between picks (min)
    "map_mode_dwell_max_s": 2.5,         # pause between picks (max)
    "map_mode_visit_radius_m": 1.5,      # cells within this of pose marked visited
    "map_mode_recency_cap_s": 600.0,     # recency score saturates here
    "map_mode_w_recency": 1.0,           # weights for the score components
    "map_mode_w_frontier": -0.8,         # negative = prefer interior cells, dont push toward unmapped edges
    "map_mode_w_distance": 0.35,
    "map_mode_max_missing_neighbors": 3, # skip cells with more missing neighbors than this (frontier guard)
    "map_mode_stale_timeout_s": 25.0,    # bail and re-pick if follow stalls
    "map_mode_waypoint_chance": 0.2,     # roll per-pick to visit a saved waypoint instead
    "map_mode_waypoint_min_dist": 2.5,   # skip waypoints closer than this (we're already there)
}
