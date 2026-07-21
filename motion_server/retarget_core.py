# ARDY Core-27 skeleton -> Gabriel FBT muscle params.
#
# ARDY world (probed 2026-07, ardy_probe_stand.npz): y-up meters, floor at
# y=0, facing +z at heading angle 0, +x is the models LEFT (right-handed).
# global_root_heading = (cos t, sin t) with positive t turning LEFT.
# Torso/leg/foot bind orientations are world aligned at stand (probe showed
# hips/spine3/foot rotmats ~identity), so the anatomical basis is literally
# the SMPL model frame: UP=+y FWD=+z LEFT=+x. Arms/hands are NOT world
# aligned at bind, so every arm/hand/leg angle is solved from joint
# positions instead of rotmats.
#
# Locomotion extras are converted to the DART convention the client already
# speaks (x=right, y=forward, yaw positive = right turn): x=-x, y=z, yaw=-t.

import math
import numpy as np

import retarget as base
from retarget import (
    PARAM_MUSCLES, NEUTRAL, SIGN, SCALE,
    HIPS_PITCH_MAX_DEG, HIPS_ROLL_MAX_DEG,
    Retargeter, _clamp,
)

# cskel27 joint indices (bone_order_names_with_parents order)
HIPS, SPINE, SPINE1, SPINE2, SPINE3, NECK, HEAD = range(7)
R_SHOULDER, R_ARM, R_FOREARM, R_HAND, R_HANDEND, R_THUMB = range(7, 13)
L_SHOULDER, L_ARM, L_FOREARM, L_HAND, L_HANDEND, L_THUMB = range(13, 19)
R_UPLEG, R_LEG, R_FOOT, R_TOE = range(19, 23)
L_UPLEG, L_LEG, L_FOOT, L_TOE = range(23, 27)

UP = np.array([0.0, 1.0, 0.0])
FWD = np.array([0.0, 0.0, 1.0])
LEFT = np.array([1.0, 0.0, 0.0])

# rest angles (radians) measured off 'a person stands still' with
# probe_core.py, subtracted so standing lands exactly on NEUTRAL.
CORE_REST = {
    'HeadNod': -0.0342,
    'HeadTilt': -0.0064,
    'HeadTurn': +0.0282,
    'HipsPitch': -0.0115,
    'HipsRoll': -0.0034,
    'LArmFB': -0.0238,
    'LArmTW': +0.8256,
    'LArmUp': +0.2106,
    'LElbow': -0.4154,
    'LFootUD': +1.2340,
    'LKnee': -0.1325,
    'LLegFB': -0.0398,
    'LLegIO': +0.1001,
    'LWristIO': -0.0612,
    'LWristUD': +0.1267,
    'RArmFB': +0.0353,
    'RArmTW': -0.8908,
    'RArmUp': +0.2034,
    'RElbow': -0.4124,
    'RFootUD': +1.1997,
    'RKnee': -0.1775,
    'RLegFB': +0.0217,
    'RLegIO': -0.1012,
    'RWristIO': -0.1378,
    'RWristUD': +0.1674,
    'SpineFB': +0.1115,
    'SpineLR': -0.0091,
    'SpineTW': -0.0335,
}

# measured stand pose: hips height and lowest joint above the y=0 floor
STAND_HIPS_Y = 0.952
STAND_FLOOR_CLEAR = 0.006


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def extract_angles_core(rotmats, joints, heading):
    """raw anatomical angles per param. rotmats [27,3,3] world, joints [27,3]
    world y-up, heading [2] = (cos t, sin t)."""
    R = rotmats
    ang = {}

    # facing frame from the models own smoothed heading (robust when lying)
    h0, h1 = float(heading[0]), float(heading[1])
    fwd_w = _norm(np.array([h1, 0.0, h0]))
    left_w = _norm(np.cross(UP, fwd_w))
    R_face = np.stack([left_w, UP, fwd_w], axis=1)  # model -> world

    ang['_yaw'] = -math.atan2(h1, h0)  # dart convention, + = right turn

    # hips lean, yaw removed
    R_hips_l = R_face.T @ R[HIPS]
    up_l = R_hips_l @ UP
    fwd_l = R_hips_l @ FWD
    ang['HipsPitch'] = math.atan2(-fwd_l[1], fwd_l[2])   # lean forward positive
    ang['HipsRoll'] = math.atan2(-up_l[0], up_l[1])      # lean right positive

    # body frame: everything below measured relative to the pelvis like the
    # smpl path, so bowing/lying keeps limb params sane
    R0T = R[HIPS].T

    def ball(R_rel, fb, lr, tw):
        d = R_rel @ UP
        ang[fb] = math.atan2(float(d @ FWD), float(d @ UP))
        ang[lr] = math.asin(_clamp(float(d @ LEFT)))
        ang[tw] = base._twist_angle(R_rel, UP)

    ball(R0T @ R[SPINE3], 'SpineFB', 'SpineLR', 'SpineTW')
    ball(R[SPINE3].T @ R[HEAD], 'HeadNod', 'HeadTilt', 'HeadTurn')

    for side, sh, el, wr, he, lat in (
            ('L', L_ARM, L_FOREARM, L_HAND, L_HANDEND, LEFT),
            ('R', R_ARM, R_FOREARM, R_HAND, R_HANDEND, -LEFT)):
        u = R0T @ _norm(joints[el] - joints[sh])   # upper arm dir, body frame
        f = R0T @ _norm(joints[wr] - joints[el])   # forearm dir

        raw_up = math.atan2(float(u @ UP), float(u @ lat)) + math.pi / 2
        coronal = math.hypot(float(u @ UP), float(u @ lat))
        w = min(1.0, coronal / 0.25)
        ang[f'{side}ArmUp'] = w * raw_up + (1.0 - w) * base.REST_RAD.get(f'{side}ArmUp', 0.0)
        ang[f'{side}ArmFB'] = math.asin(_clamp(float(u @ FWD)))

        bend = math.acos(_clamp(float(u @ f)))
        ang[f'{side}Elbow'] = -bend

        ref = FWD - float(FWD @ u) * u
        ref_n = np.linalg.norm(ref)
        f_perp = f - float(f @ u) * u
        f_n = np.linalg.norm(f_perp)
        if ref_n > 1e-6 and f_n > 1e-6:
            ref = ref / ref_n
            f_perp = f_perp / f_n
            tw_raw = math.atan2(float(f_perp @ np.cross(u, ref)), float(f_perp @ ref))
        else:
            tw_raw = base.REST_RAD.get(f'{side}ArmTW', 0.0)
        wt = min(1.0, bend / 0.15)
        ang[f'{side}ArmTW'] = wt * tw_raw + (1.0 - wt) * base.REST_RAD.get(f'{side}ArmTW', 0.0)

        # hand pointing dir in the forearm frame (bind not world aligned,
        # rest calibration mops up the constant offset)
        d_hand = R[el].T @ _norm(joints[he] - joints[wr])
        ang[f'{side}WristUD'] = math.asin(_clamp(float(d_hand @ UP)))
        ang[f'{side}WristIO'] = math.atan2(float(d_hand @ FWD), float(d_hand @ lat))

    for side, hip, knee, ankle, toe in (('L', L_UPLEG, L_LEG, L_FOOT, L_TOE),
                                        ('R', R_UPLEG, R_LEG, R_FOOT, R_TOE)):
        d = R0T @ _norm(joints[knee] - joints[hip])       # thigh dir
        s = R0T @ _norm(joints[ankle] - joints[knee])     # shin dir
        ang[f'{side}LegFB'] = math.atan2(float(d @ FWD), float(-d @ UP))
        ang[f'{side}LegIO'] = math.asin(_clamp(float(d @ LEFT)))
        ang[f'{side}Knee'] = -math.acos(_clamp(float(d @ s)))
        d_foot = R0T @ _norm(joints[toe] - joints[ankle])
        ang[f'{side}FootUD'] = math.atan2(float(d_foot @ FWD), float(-d_foot @ UP))

    return ang


class CoreRetargeter(Retargeter):
    """same muscle gain machinery, ardy core27 inputs."""

    def __init__(self, ranges_path, fps=20):
        base.REST_PRESETS['core'] = CORE_REST
        base.set_rest('core')
        super().__init__(ranges_path, fps=fps)

    def frame_to_params(self, joints, rotmats, heading, root_pos, smooth_root):
        """joints [27,3], rotmats [27,3,3], heading [2], root_pos [3],
        smooth_root [3], all world y-up numpy."""
        ang = extract_angles_core(rotmats, joints, heading)
        out = {}
        for param in PARAM_MUSCLES:
            out[param] = self._map(param, ang[param])

        out['HipsPitch'] = _clamp(
            math.degrees(ang['HipsPitch'] - base.REST_RAD.get('HipsPitch', 0.0)) / HIPS_PITCH_MAX_DEG)
        out['HipsRoll'] = _clamp(
            math.degrees(ang['HipsRoll'] - base.REST_RAD.get('HipsRoll', 0.0)) / HIPS_ROLL_MAX_DEG)

        # floor is y=0 exactly; only ever shift up (jumps stay real)
        ground_shift = max(0.0, STAND_FLOOR_CLEAR - float(joints[:, 1].min()))
        dz = (float(root_pos[1]) + ground_shift) - STAND_HIPS_Y
        out['HipsY'] = _clamp(dz / self.hipsy_up_m if dz >= 0 else dz / self.hipsy_down_m)

        # locomotion extras in the dart convention (x=right, y=fwd, yaw +=right)
        x, y, yaw = -float(smooth_root[0]), float(smooth_root[2]), ang['_yaw']
        out['_yaw'] = yaw
        out['_x'] = x
        out['_y'] = y
        if self._prev_root is not None:
            px, py, pyaw = self._prev_root
            dt = 1.0 / self.fps
            dx, dy = x - px, y - py
            s, c = math.sin(yaw), math.cos(yaw)
            out['_vfwd'] = (dx * s + dy * c) / dt
            out['_vside'] = (dx * c - dy * s) / dt
            dyaw = (yaw - pyaw + math.pi) % (2.0 * math.pi) - math.pi
            out['_vyaw'] = dyaw / dt
        else:
            out['_vfwd'] = out['_vside'] = out['_vyaw'] = 0.0
        self._prev_root = (x, y, yaw)
        return out
