# SMPL-X world frames -> Gabriel FBT muscle params.
# DART world is z-up (x right, y forward at canonical facing).
# smpl local joint frames are axis aligned with the model frame at rest,
# so swing/twist decomposition against rest bone directions works everywhere.
#
# unity muscle 0 is NOT anatomical rest (thighs 29 deg forward, arms raised
# etc), so each param is anchored at the NEUTRAL value that was verified
# in game to give a natural standing pose, and smpl angle deltas from
# anatomical rest are applied around that anchor in muscle-degree space.
#
# SIGN/SCALE are the empirical tuning knobs, fixed via the unity sampler.

import json
import math
import numpy as np

# smplx joint indices (first 22, body only)
PELVIS, L_HIP, R_HIP, SPINE1, L_KNEE, R_KNEE, SPINE2, L_ANKLE, R_ANKLE, SPINE3, \
    L_FOOT, R_FOOT, NECK, L_COLLAR, R_COLLAR, HEAD, L_SHOULDER, R_SHOULDER, \
    L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = range(22)

# model-frame basis at rest (y-up smpl convention)
UP = np.array([0.0, 1.0, 0.0])
FWD = np.array([0.0, 0.0, 1.0])
LEFT = np.array([1.0, 0.0, 0.0])

BONE_DIR = {
    'l_arm': LEFT,
    'r_arm': -LEFT,
    'leg': -UP,
    'spine': UP,
}

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

# param values for the anatomical standing rest pose (verified in game)
NEUTRAL = {
    'SpineFB': 0.0, 'SpineLR': 0.0, 'SpineTW': 0.0,
    'HeadNod': 0.0, 'HeadTilt': 0.0, 'HeadTurn': 0.0,
    'LArmUp': -0.75, 'LArmFB': 0.0, 'LArmTW': 0.0,
    'LElbow': 0.85, 'LWristUD': 0.0, 'LWristIO': 0.0,
    'RArmUp': -0.75, 'RArmFB': 0.0, 'RArmTW': 0.0,
    'RElbow': 0.85, 'RWristUD': 0.0, 'RWristIO': 0.0,
    'LLegFB': 0.58, 'LLegIO': 0.0, 'LKnee': 0.85, 'LFootUD': -0.3,
    'RLegFB': 0.58, 'RLegIO': 0.0, 'RKnee': 0.85, 'RFootUD': -0.3,
}

# empirical direction/scale corrections, tuned via the unity sampler.
# unity muscle naming: the FIRST word is the negative end ("Front-Back" means
# -1=front +1=back, "Down-Up" means -1=down, "In-Out" means -1=in). our
# anatomical angles are positive toward front/left/up, hence the flips.
SIGN = {p: 1.0 for p in PARAM_MUSCLES}
SIGN.update({
    'SpineFB': -1.0, 'SpineLR': -1.0, 'SpineTW': -1.0,
    'HeadNod': -1.0, 'HeadTilt': -1.0, 'HeadTurn': -1.0,
    'LArmFB': -1.0, 'RArmFB': -1.0,
    'LArmTW': -1.0,  # twist axes mirror between sides
    'LWristIO': -1.0, 'RWristIO': -1.0,
    'LFootUD': -1.0, 'RFootUD': -1.0,
})
SCALE = {p: 1.0 for p in PARAM_MUSCLES}

HIPS_PITCH_MAX_DEG = 40.0
HIPS_ROLL_MAX_DEG = 30.0
# smpl standing pelvis carries a slight backward tilt, measured off the stand prompt
HIPS_PITCH_REST_DEG = -3.8


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


class Retargeter:
    def __init__(self, ranges_path):
        with open(ranges_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.muscles = data['muscles']  # name -> {min, max} degrees
        self.human_scale = data.get('humanScale', 1.0)
        # dart world origin sits at the initial pelvis, so standing pelvis z is 0
        self.smpl_stand_z = data.get('smplStandZ', 0.0)
        self.hipsy_down_m = data.get('hipsYDownMeters', 0.65)
        self.hipsy_up_m = data.get('hipsYUpMeters', 0.14)

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

    def _map(self, param, delta_rad):
        """anatomical delta from standing rest -> param value."""
        theta = self.theta_neutral[param] + SIGN[param] * SCALE[param] * delta_rad
        gain = self.pos_gain[param] if theta >= 0 else self.neg_gain[param]
        if gain < 1e-6:
            return 0.0
        return _clamp(theta / gain)

    def frame_to_params(self, transl, rotmats, joints):
        """transl [3], rotmats [22,3,3] (0=global orient world zup, 1..21 local), joints [22,3] world."""
        R = rotmats
        out = {}

        # --- root ---
        up_w = R[0] @ UP
        fwd_w = R[0] @ FWD
        yaw = math.atan2(fwd_w[0], fwd_w[1])  # facing about world z
        # fwd_w = Rz(-yaw) @ +y, so applying Rz(+yaw) cancels the facing
        cy, sy = math.cos(yaw), math.sin(yaw)
        yaw_inv = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        up_l = yaw_inv @ up_w
        fwd_l = yaw_inv @ fwd_w
        pitch = math.atan2(-fwd_l[2], fwd_l[1])  # lean forward positive
        roll = math.atan2(up_l[0], up_l[2])      # lean right positive
        out['HipsPitch'] = _clamp((math.degrees(pitch) - HIPS_PITCH_REST_DEG) / HIPS_PITCH_MAX_DEG)
        out['HipsRoll'] = _clamp(math.degrees(roll) / HIPS_ROLL_MAX_DEG)
        # smplx transl is the root offset, not pelvis position. use the
        # network predicted pelvis joint for actual height.
        dz = float(joints[PELVIS][2]) - self.smpl_stand_z
        out['HipsY'] = _clamp(dz / self.hipsy_up_m if dz >= 0 else dz / self.hipsy_down_m)
        # extras for the client locomotion layer (not muscle params)
        out['_yaw'] = yaw
        out['_x'] = float(joints[PELVIS][0])
        out['_y'] = float(joints[PELVIS][1])

        # --- spine / head ---
        self._ball(out, R[SPINE1] @ R[SPINE2] @ R[SPINE3], 'SpineFB', 'SpineLR', 'SpineTW')
        self._ball(out, R[NECK] @ R[HEAD], 'HeadNod', 'HeadTilt', 'HeadTurn')

        # --- arms ---
        for side, collar, shoulder, elbow, wrist in (
                ('L', L_COLLAR, L_SHOULDER, L_ELBOW, L_WRIST),
                ('R', R_COLLAR, R_SHOULDER, R_ELBOW, R_WRIST)):
            lat = BONE_DIR[f'{side.lower()}_arm']
            R_arm = R[collar] @ R[shoulder]
            d = R_arm @ lat
            # standing rest: arm hangs down = -90deg elevation vs t-pose
            elev = math.asin(_clamp(float(d @ UP)))
            out[f'{side}ArmUp'] = self._map(f'{side}ArmUp', elev + math.pi / 2)
            horiz = math.atan2(float(d @ FWD), float(d @ lat))
            out[f'{side}ArmFB'] = self._map(f'{side}ArmFB', horiz)
            out[f'{side}ArmTW'] = self._map(f'{side}ArmTW', _twist_angle(R_arm, lat))

            bend = math.acos(_clamp(float((R[elbow] @ lat) @ lat)))  # 0 = straight
            out[f'{side}Elbow'] = self._map(f'{side}Elbow', -bend)

            d_hand = R[wrist] @ lat
            out[f'{side}WristUD'] = self._map(f'{side}WristUD', math.asin(_clamp(float(d_hand @ UP))))
            out[f'{side}WristIO'] = self._map(f'{side}WristIO', math.atan2(float(d_hand @ FWD), float(d_hand @ lat)))

        # --- legs ---
        for side, hip, knee, ankle in (('L', L_HIP, L_KNEE, L_ANKLE),
                                       ('R', R_HIP, R_KNEE, R_ANKLE)):
            d = R[hip] @ BONE_DIR['leg']
            fb = math.atan2(float(d @ FWD), float(-d @ UP))  # thigh forward positive
            io = math.asin(_clamp(float(d @ LEFT)))
            # unity leg Front-Back muscle: negative side = forward swing
            out[f'{side}LegFB'] = self._map(f'{side}LegFB', -fb)
            out[f'{side}LegIO'] = self._map(f'{side}LegIO', io if side == 'L' else -io)

            bend = math.acos(_clamp(float((R[knee] @ BONE_DIR['leg']) @ BONE_DIR['leg'])))
            out[f'{side}Knee'] = self._map(f'{side}Knee', -bend)

            d_foot = R[ankle] @ BONE_DIR['leg']
            out[f'{side}FootUD'] = self._map(f'{side}FootUD', math.atan2(float(d_foot @ FWD), float(-d_foot @ UP)))

        return out

    def _ball(self, out, R_agg, fb, lr, tw):
        d = R_agg @ UP
        out[fb] = self._map(fb, math.atan2(float(d @ FWD), float(d @ UP)))
        out[lr] = self._map(lr, math.asin(_clamp(float(d @ LEFT))))
        out[tw] = self._map(tw, _twist_angle(R_agg, UP))
