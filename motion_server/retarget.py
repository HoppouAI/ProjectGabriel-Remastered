# SMPL-X world frames -> Gabriel FBT muscle params.
# DART world is z-up (x right, y forward at canonical facing).
# smpl local joint frames are axis aligned with the model frame at rest,
# so swing/twist decomposition against rest bone directions works everywhere.
#
# mapping: raw anatomical angle -> subtract REST_RAD (measured off DART's
# 'stand' prompt so standing lands exactly on NEUTRAL) -> SIGN/SCALE ->
# add on top of the NEUTRAL anchor in muscle-degree space -> param.
#
# unity muscle 0 is NOT anatomical rest (thighs 29 deg forward, arms raised
# etc), NEUTRAL is the in-game verified standing pose that compensates.

import json
import math
import numpy as np

# smplx joint indices (first 22, body only)
PELVIS, L_HIP, R_HIP, SPINE1, L_KNEE, R_KNEE, SPINE2, L_ANKLE, R_ANKLE, SPINE3, \
    L_FOOT, R_FOOT, NECK, L_COLLAR, R_COLLAR, HEAD, L_SHOULDER, R_SHOULDER, \
    L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = range(22)

# model-frame basis at rest (y-up smpl convention, +x = model left)
UP = np.array([0.0, 1.0, 0.0])
FWD = np.array([0.0, 0.0, 1.0])
LEFT = np.array([1.0, 0.0, 0.0])

# param -> [(unity muscle name, dbt weight)], mirrors the DesktopFBT blend tree
PARAM_MUSCLES = {
    'SpineFB': [('Spine Front-Back', 1.0), ('Chest Front-Back', 0.7), ('UpperChest Front-Back', 0.5)],
    'SpineLR': [('Spine Left-Right', 1.0), ('Chest Left-Right', 0.7), ('UpperChest Left-Right', 0.5)],
    'SpineTW': [('Spine Twist Left-Right', 1.0), ('Chest Twist Left-Right', 0.7), ('UpperChest Twist Left-Right', 0.5)],
    'HeadNod': [('Neck Nod Down-Up', 0.6), ('Head Nod Down-Up', 1.0)],
    'HeadTilt': [('Neck Tilt Left-Right', 0.6), ('Head Tilt Left-Right', 1.0)],
    'HeadTurn': [('Neck Turn Left-Right', 0.6), ('Head Turn Left-Right', 1.0)],
    'LArmUp': [('Left Shoulder Down-Up', 0.5), ('Left Arm Down-Up', 1.0)],
    'LArmFB': [('Left Shoulder Front-Back', 0.5), ('Left Arm Front-Back', 1.0)],
    'LArmTW': [('Left Arm Twist In-Out', 1.0)],
    'LElbow': [('Left Forearm Stretch', 1.0)],
    'LWristUD': [('Left Hand Down-Up', 1.0)],
    'LWristIO': [('Left Hand In-Out', 1.0)],
    'RArmUp': [('Right Shoulder Down-Up', 0.5), ('Right Arm Down-Up', 1.0)],
    'RArmFB': [('Right Shoulder Front-Back', 0.5), ('Right Arm Front-Back', 1.0)],
    'RArmTW': [('Right Arm Twist In-Out', 1.0)],
    'RElbow': [('Right Forearm Stretch', 1.0)],
    'RWristUD': [('Right Hand Down-Up', 1.0)],
    'RWristIO': [('Right Hand In-Out', 1.0)],
    'LLegFB': [('Left Upper Leg Front-Back', 1.0)],
    'LLegIO': [('Left Upper Leg In-Out', 1.0)],
    'LKnee': [('Left Lower Leg Stretch', 1.0)],
    'LFootUD': [('Left Foot Up-Down', 1.0)],
    'RLegFB': [('Right Upper Leg Front-Back', 1.0)],
    'RLegIO': [('Right Upper Leg In-Out', 1.0)],
    'RKnee': [('Right Lower Leg Stretch', 1.0)],
    'RFootUD': [('Right Foot Up-Down', 1.0)],
}

# param values for the anatomical standing rest pose. arm values solved in
# the unity sampler so the wrists land where darts stand pose puts them
# (hands by the outer thighs, ~0.29m lateral), not the old hand-tuned guess
# that parked them in front of the crotch. ArmTW 0.4 rotates the elbow
# flexion plane outward which is what actually frees the hands.
NEUTRAL = {
    'SpineFB': 0.0, 'SpineLR': 0.0, 'SpineTW': 0.0,
    'HeadNod': 0.0, 'HeadTilt': 0.0, 'HeadTurn': 0.0,
    'LArmUp': -0.6, 'LArmFB': 0.1, 'LArmTW': 0.4,
    'LElbow': 1.0, 'LWristUD': 0.0, 'LWristIO': 0.0,
    'RArmUp': -0.6, 'RArmFB': 0.1, 'RArmTW': 0.4,
    'RElbow': 1.0, 'RWristUD': 0.0, 'RWristIO': 0.0,
    'LLegFB': 0.58, 'LLegIO': 0.0, 'LKnee': 0.85, 'LFootUD': -0.3,
    'RLegFB': 0.58, 'RLegIO': 0.0, 'RKnee': 0.85, 'RFootUD': -0.3,
}

# rest angles (radians) of DART's 'stand' prompt in the raw anatomical
# convention below. subtracted so standing maps exactly to NEUTRAL.
# regenerate with probe_axes.py after model/prompt changes.
REST_RAD = {
    'HeadNod': -0.0351,
    'HeadTilt': -0.0935,
    'HeadTurn': -0.0984,
    'HipsPitch': -0.1169,
    'HipsRoll': -0.0220,
    'LArmFB': -0.2782,
    'LArmTW': +0.0301,
    'LArmUp': +0.1336,
    'LElbow': -0.6319,
    'LFootUD': +0.0915,
    'LKnee': -0.1592,
    'LLegFB': -0.1898,
    'LLegIO': -0.1065,
    'LWristIO': +0.1369,
    'LWristUD': -0.0044,
    'RArmFB': -0.2133,
    'RArmTW': -0.1597,
    'RArmUp': +0.1766,
    'RElbow': -0.4371,
    'RFootUD': -0.0044,
    'RKnee': -0.1559,
    'RLegFB': -0.1528,
    'RLegIO': +0.1262,
    'RWristIO': +0.0732,
    'RWristUD': -0.0895,
    'SpineFB': +0.1831,
    'SpineLR': -0.0329,
    'SpineTW': -0.0516,
}

# raw anatomical convention: forward/left/up positive, twist by right hand
# rule about the rest bone axis. unity muscle naming has the FIRST word as
# the negative end, hence the flips.
SIGN = {p: 1.0 for p in PARAM_MUSCLES}
SIGN.update({
    'SpineFB': -1.0, 'SpineLR': -1.0, 'SpineTW': -1.0,
    'HeadNod': -1.0, 'HeadTilt': -1.0, 'HeadTurn': -1.0,
    'LArmFB': -1.0, 'RArmFB': -1.0,
    'LArmTW': -1.0,  # twist axes mirror between sides
    'LWristIO': -1.0, 'RWristIO': -1.0,
    'LLegFB': -1.0, 'RLegFB': -1.0,
    'RLegIO': -1.0,  # abduction measured toward +x, mirrors for the right leg
    'LFootUD': -1.0, 'RFootUD': -1.0,
})
SCALE = {p: 1.0 for p in PARAM_MUSCLES}

HIPS_PITCH_MAX_DEG = 40.0
HIPS_ROLL_MAX_DEG = 30.0


def _twist_angle(R, axis):
    """signed rotation of R about unit axis (radians), via quaternion projection."""
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w < 1e-8:
        return 0.0
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    proj = x * axis[0] + y * axis[1] + z * axis[2]
    return 2.0 * math.atan2(proj, w)


def _clamp(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))


def extract_angles(rotmats, joints):
    """raw anatomical angles (radians) per param plus root pitch/roll/yaw.
    arms are solved from predicted joint positions (shoulder->elbow->wrist)
    so the hands end up where the model put them, rotation chains only
    supply the wrist orientation and everything below the hips."""
    R = rotmats
    ang = {}

    up_w = R[0] @ UP
    fwd_w = R[0] @ FWD
    yaw = math.atan2(fwd_w[0], fwd_w[1])  # facing about world z
    # fwd_w = Rz(-yaw) @ +y, so applying Rz(+yaw) cancels the facing
    cy, sy = math.cos(yaw), math.sin(yaw)
    yaw_inv = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    up_l = yaw_inv @ up_w
    fwd_l = yaw_inv @ fwd_w
    ang['_yaw'] = yaw
    ang['HipsPitch'] = math.atan2(-fwd_l[2], fwd_l[1])  # lean forward positive
    ang['HipsRoll'] = math.atan2(up_l[0], up_l[2])      # lean right positive

    def ball(R_agg, fb, lr, tw):
        d = R_agg @ UP
        ang[fb] = math.atan2(float(d @ FWD), float(d @ UP))
        ang[lr] = math.asin(_clamp(float(d @ LEFT)))
        ang[tw] = _twist_angle(R_agg, UP)

    ball(R[SPINE1] @ R[SPINE2] @ R[SPINE3], 'SpineFB', 'SpineLR', 'SpineTW')
    ball(R[NECK] @ R[HEAD], 'HeadNod', 'HeadTilt', 'HeadTurn')

    # body frame: rows of R0^T map world -> model space
    R0T = R[0].T

    def norm(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else v

    for side, sh, el, wr, wrist_j, lat in (
            ('L', L_SHOULDER, L_ELBOW, L_WRIST, L_WRIST, LEFT),
            ('R', R_SHOULDER, R_ELBOW, R_WRIST, R_WRIST, -LEFT)):
        u = R0T @ norm(joints[el] - joints[sh])   # upper arm dir, model frame
        f = R0T @ norm(joints[wr] - joints[el])   # forearm dir, model frame

        # coronal plane angle: hanging = 0 (after +90 shift), t-pose = +90,
        # adduction goes negative, continuous through a wave. pole only at a
        # pure forward point, held at rest there.
        raw_up = math.atan2(float(u @ UP), float(u @ lat)) + math.pi / 2
        coronal = math.hypot(float(u @ UP), float(u @ lat))
        w = min(1.0, coronal / 0.25)
        ang[f'{side}ArmUp'] = w * raw_up + (1.0 - w) * REST_RAD.get(f'{side}ArmUp', 0.0)
        ang[f'{side}ArmFB'] = math.asin(_clamp(float(u @ FWD)))

        # elbow bend straight from the joint angle
        bend = math.acos(_clamp(float(u @ f)))
        ang[f'{side}Elbow'] = -bend

        # twist = azimuth of the forearm around the upper arm axis, measured
        # from the forward flexion plane (unity muscle 0 bends forward).
        # undefined for a straight arm, blend to rest as the bend vanishes.
        ref = FWD - float(FWD @ u) * u
        ref_n = np.linalg.norm(ref)
        f_perp = f - float(f @ u) * u
        f_n = np.linalg.norm(f_perp)
        if ref_n > 1e-6 and f_n > 1e-6:
            ref = ref / ref_n
            f_perp = f_perp / f_n
            tw_raw = math.atan2(float(f_perp @ np.cross(u, ref)), float(f_perp @ ref))
        else:
            tw_raw = REST_RAD.get(f'{side}ArmTW', 0.0)
        wt = min(1.0, bend / 0.15)
        ang[f'{side}ArmTW'] = wt * tw_raw + (1.0 - wt) * REST_RAD.get(f'{side}ArmTW', 0.0)

        d_hand = R[wrist_j] @ lat
        ang[f'{side}WristUD'] = math.asin(_clamp(float(d_hand @ UP)))
        ang[f'{side}WristIO'] = math.atan2(float(d_hand @ FWD), float(d_hand @ lat))

    down = -UP
    for side, hip, knee, ankle in (('L', L_HIP, L_KNEE, L_ANKLE),
                                   ('R', R_HIP, R_KNEE, R_ANKLE)):
        d = R[hip] @ down
        ang[f'{side}LegFB'] = math.atan2(float(d @ FWD), float(-d @ UP))  # thigh forward positive
        ang[f'{side}LegIO'] = math.asin(_clamp(float(d @ LEFT)))          # toward +x positive
        ang[f'{side}Knee'] = -math.acos(_clamp(float((R[knee] @ down) @ down)))
        d_foot = R[ankle] @ down
        ang[f'{side}FootUD'] = math.atan2(float(d_foot @ FWD), float(-d_foot @ UP))

    return ang


class Retargeter:
    def __init__(self, ranges_path):
        with open(ranges_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.muscles = data['muscles']  # name -> {min, max} degrees
        self.human_scale = data.get('humanScale', 1.0)
        # dart world floor is z=0 with the seed standing on it
        self.smpl_stand_z = data.get('smplStandZ', 0.0)
        self.hipsy_down_m = data.get('hipsYDownMeters', 0.65)
        self.hipsy_up_m = data.get('hipsYUpMeters', 0.14)
        self._prev_root = None  # (x, y, yaw) of the previous frame

        # chain gain (radians of joint motion per unit of param) each side of 0
        self.pos_gain = {}
        self.neg_gain = {}
        self.theta_neutral = {}
        for param, entries in PARAM_MUSCLES.items():
            pos = neg = 0.0
            for name, w in entries:
                r = self.muscles.get(name)
                if r is None:
                    raise KeyError(f'muscle missing from ranges dump: {name}')
                pos += w * r['max']
                neg += w * abs(r['min'])
            self.pos_gain[param] = math.radians(pos)
            self.neg_gain[param] = math.radians(neg)
            n = NEUTRAL.get(param, 0.0)
            self.theta_neutral[param] = n * (self.pos_gain[param] if n >= 0 else self.neg_gain[param])

    def reset_root(self):
        """forget the previous root sample so velocities dont spike after an engine reset."""
        self._prev_root = None

    def _map(self, param, raw_rad):
        delta = raw_rad - REST_RAD.get(param, 0.0)
        theta = self.theta_neutral[param] + SIGN[param] * SCALE[param] * delta
        gain = self.pos_gain[param] if theta >= 0 else self.neg_gain[param]
        if gain < 1e-6:
            return 0.0
        return _clamp(theta / gain)

    def frame_to_params(self, transl, rotmats, joints):
        """transl [3], rotmats [22,3,3] (0=global orient world zup, 1..21 local), joints [22,3] world."""
        ang = extract_angles(rotmats, joints)
        out = {}
        for param in PARAM_MUSCLES:
            out[param] = self._map(param, ang[param])

        out['HipsPitch'] = _clamp(
            math.degrees(ang['HipsPitch'] - REST_RAD.get('HipsPitch', 0.0)) / HIPS_PITCH_MAX_DEG)
        out['HipsRoll'] = _clamp(
            math.degrees(ang['HipsRoll'] - REST_RAD.get('HipsRoll', 0.0)) / HIPS_ROLL_MAX_DEG)

        # ground clamp: rollout drift can sink the whole body below the dart
        # floor and it never recovers. the floor plane sits where the standing
        # feet are (min joint z ~= -0.984, origin is at the seed pelvis).
        # shift up so the lowest joint stays at/above the floor, never push
        # down (jumps stay real).
        floor_z = self.smpl_stand_z - 0.984
        ground_shift = max(0.0, floor_z - float(joints[:, 2].min()))
        pelvis_z = float(joints[PELVIS][2]) + ground_shift
        dz = pelvis_z - self.smpl_stand_z
        out['HipsY'] = _clamp(dz / self.hipsy_up_m if dz >= 0 else dz / self.hipsy_down_m)

        # extras for the client locomotion layer (not muscle params)
        x, y, yaw = float(joints[PELVIS][0]), float(joints[PELVIS][1]), ang['_yaw']
        out['_yaw'] = yaw
        out['_x'] = x
        out['_y'] = y
        # body frame velocities at 30fps: forward, rightward (m/s), yaw rate
        # (rad/s, positive = turn right)
        if self._prev_root is not None:
            px, py, pyaw = self._prev_root
            dt = 1.0 / 30.0
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
