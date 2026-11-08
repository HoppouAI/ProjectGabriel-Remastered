# solves fbt params against tables measured on the real avatar (Tools > ProjectGabriel > Export FBT calibration)

import json
import math
from itertools import product

import numpy as np

ARM = ('ArmUp', 'ArmFB', 'ArmTW')
LEG = ('LegFB', 'LegIO')
SPINE = ('SpineFB', 'SpineLR', 'SpineTW')
HEAD = ('HeadNod', 'HeadTilt', 'HeadTurn')

TWIST_FULL_BEND = 0.35   # rad of elbow bend before the twist reading is fully trusted
PRIOR = 0.02             # pull toward last frames answer, keeps ambiguous joints from hopping
GN_ITERS = 6
FD_H = 1e-3


def _norm(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def _perp(f, u):
    return _norm(f - np.sum(f * u, axis=-1, keepdims=True) * u)


def _quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _mat_to_quat(m):
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


class _Grid:
    """multilinear lookup over [-1,1]^d with values of size k per node."""

    def __init__(self, values, n, d):
        self.v = np.asarray(values, dtype=np.float64).reshape((n,) * d + (-1,))
        self.n, self.d = n, d
        self.nodes = np.stack(np.meshgrid(*[np.linspace(-1, 1, n)] * d, indexing='ij'), -1).reshape(-1, d)
        self.flat = self.v.reshape(-1, self.v.shape[-1])
        self._corners = np.array(list(product((0, 1), repeat=d)))

    def __call__(self, p):
        """p is (d,) or (m, d), returns (k,) or (m, k)."""
        p = np.asarray(p, dtype=np.float64)
        single = p.ndim == 1
        p = np.atleast_2d(p)
        t = (np.clip(p, -1, 1) + 1) * 0.5 * (self.n - 1)
        i0 = np.clip(np.floor(t).astype(int), 0, self.n - 2)
        w = t - i0
        idx = i0[:, None, :] + self._corners[None]
        f = np.where(self._corners[None] == 1, w[:, None, :], 1.0 - w[:, None, :])
        wt = np.prod(f, axis=-1)
        vals = self.v[tuple(np.moveaxis(idx, -1, 0))]
        out = np.einsum('mc,mck->mk', wt, vals)
        return out[0] if single else out


class _Table1D:
    """monotonic joint angle table, inverted by interpolation."""

    def __init__(self, values):
        a = np.asarray(values, dtype=np.float64)
        p = np.linspace(-1, 1, len(a))
        if a[0] > a[-1]:
            a, p = a[::-1], p[::-1]
        self.a, self.p = a, p

    def solve(self, angle):
        return float(np.interp(angle, self.a, self.p))


def _gauss_newton(feat_batch, target, x0, weights, prior_to=None, prior_w=0.0):
    """feat_batch maps (m, d) params to (m, k) features, one call per iteration."""
    x = np.array(x0, dtype=np.float64)
    d = len(x)
    offs = np.vstack([np.zeros(d), FD_H * np.eye(d), -FD_H * np.eye(d)])
    for _ in range(GN_ITERS):
        F = (feat_batch(x[None] + offs) - target) * weights
        r = F[0]
        J = ((F[1:d + 1] - F[d + 1:]) / (2 * FD_H)).T
        A = J.T @ J + (1e-4 + prior_w) * np.eye(d)
        g = J.T @ r
        if prior_to is not None and prior_w:
            g = g + prior_w * (x - prior_to)
        step = np.linalg.solve(A, -g)
        x = np.clip(x + step, -1, 1)
        if np.max(np.abs(step)) < 1e-4:
            break
    return x


def _cost(r):
    return float(np.sum(r * r))


GOOD_ENOUGH = 1e-4  # skip the table search when the warm start already lands this close


class FBTCalibration:
    def __init__(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            c = json.load(f)
        if c.get('version') != 1:
            raise ValueError(f'unsupported calibration version {c.get("version")}')
        g = c['grids']
        self.avatar = c.get('avatar', '?')
        self.arm, self.arm_fp, self.leg = {}, {}, {}
        self.elbow, self.knee, self.foot = {}, {}, {}
        for s in 'LR':
            u = np.asarray(c[f'arm{s}']['u'], dtype=np.float64)
            f = np.asarray(c[f'arm{s}']['f'], dtype=np.float64)
            self.arm[s] = _Grid(np.concatenate([u, _perp(f, u)], axis=-1), g['arm'], 3)
            self.leg[s] = _Grid(c[f'leg{s}'], g['leg'], 2)
            self.elbow[s] = _Table1D(c[f'elbow{s}'])
            self.knee[s] = _Table1D(c[f'knee{s}'])
            self.foot[s] = _Table1D(c[f'foot{s}'])
        self.spine = _Grid(c['spine'], g['rot'], 3)
        self.head = _Grid(c['head'], g['rot'], 3)
        self._prev = {}

    def reset(self):
        self._prev = {}

    def _pick(self, key, grid, feat_of_flat, target, weights, x_default):
        """best of a warm start and the closest table node, then refine."""
        prev = self._prev.get(key)
        pw = PRIOR if prev is not None else 0.0

        def feat_batch(P):
            return feat_of_flat(grid(P))

        def refine(x0):
            x = _gauss_newton(feat_batch, target, x0, weights, prior_to=prev, prior_w=pw)
            r = (feat_batch(x[None])[0] - target) * weights
            return x, _cost(r), _cost(r) + (pw * float(np.sum((x - prev) ** 2)) if pw else 0.0)

        best, fit, best_c = refine(np.asarray(prev if prev is not None else x_default, dtype=np.float64))
        if prev is None or fit > GOOD_ENOUGH:
            r_all = (feat_of_flat(grid.flat) - target) * weights
            cost = np.sum(r_all * r_all, axis=-1)
            if pw:
                cost = cost + pw * np.sum((grid.nodes - prev) ** 2, axis=-1)
            x, _, c = refine(grid.nodes[int(np.argmin(cost))])
            if c < best_c:
                best = x
        self._prev[key] = best
        return best

    def solve_arm(self, side, u, f, neutral_tw):
        u, f = _norm(np.asarray(u, float)), _norm(np.asarray(f, float))
        bend = math.acos(max(-1.0, min(1.0, float(u @ f))))
        wt = min(1.0, bend / TWIST_FULL_BEND)
        fp = _perp(f, u) if wt > 0 else np.zeros(3)
        target = np.concatenate([u, fp])
        weights = np.array([1, 1, 1, wt, wt, wt], dtype=np.float64)

        def feat(v):
            return np.concatenate([_norm(v[..., :3]), _perp(v[..., 3:], _norm(v[..., :3]))], axis=-1)

        key = 'arm' + side
        if wt < 1.0:
            # straight arm hides the twist, drift it toward the tuned neutral instead of holding junk
            prev = self._prev.get(key)
            if prev is not None:
                prev = prev.copy()
                prev[2] += (neutral_tw - prev[2]) * (1.0 - wt) * 0.1
                self._prev[key] = prev
        x = self._pick(key, self.arm[side], feat, target, weights, (0.0, 0.0, neutral_tw))
        elbow = self.elbow[side].solve(bend)
        return {f'{side}ArmUp': x[0], f'{side}ArmFB': x[1], f'{side}ArmTW': x[2], f'{side}Elbow': elbow}

    def solve_leg(self, side, thigh, shin, foot):
        thigh, shin, foot = (_norm(np.asarray(v, float)) for v in (thigh, shin, foot))
        x = self._pick('leg' + side, self.leg[side], _norm, thigh, np.ones(3), (0.6, 0.0))
        knee = math.acos(max(-1.0, min(1.0, float(thigh @ shin))))
        ankle = math.acos(max(-1.0, min(1.0, float(shin @ foot))))
        return {f'{side}LegFB': x[0], f'{side}LegIO': x[1],
                f'{side}Knee': self.knee[side].solve(knee), f'{side}FootUD': self.foot[side].solve(ankle)}

    def _solve_rot(self, key, grid, names, R):
        q = _mat_to_quat(R)
        if q[3] < 0:
            q = -q

        def feat(v):
            v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)
            return v * np.sign(v[..., 3:4] + 1e-12)

        x = self._pick(key, grid, feat, q, np.ones(4), (0.0, 0.0, 0.0))
        return dict(zip(names, x))

    def solve_spine(self, R_rel):
        return self._solve_rot('spine', self.spine, SPINE, R_rel)

    def solve_head(self, R_rel):
        return self._solve_rot('head', self.head, HEAD, R_rel)
