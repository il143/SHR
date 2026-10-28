# -*- coding: utf-8 -*-
"""SHR structural probes.

- Probes run in parallel; each structural attribute gets a 0-1 confidence
  and attributes are not mutually exclusive.
- Probes only do low-cost deterministic measurements; they do not build
  feature pools.
- A pluggable PROBES registry is traversed by the main flow.
- Key probes run on both train/val halves and take the min confidence
  (to avoid single-split overfitting).

Each probe returns an Observation:
  attr        attribute name
  confidence  0-1
  col         associated column (None for additive/global probes)
  evidence    actionable peel dict (None = signal only)
  detail      extra info (power/grouping/frequency/phase)
"""
import numpy as np
from shr_surface import Surface
from auto_decompose_v3 import _mult_err
import shr_collapse as SC

# 探针用快速 Surface（只需结构信号，不追求高保真）
# 3000 epochs 是让 sweep 曲面光滑、mult 硬判据置信度恢复 >=0.9 的最小点
PROBE_EPOCHS = 3000


def clamp(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))


# ================= State（当前问题状态） =================
class State:
    def __init__(self, X, y, col_expr, names, logy=None, seed=0,
                 epochs=PROBE_EPOCHS):
        self.X = np.asarray(X, float)
        self.y = np.asarray(y, float)
        self.col_expr = list(col_expr)
        self.names = list(names)
        self.epochs = epochs
        same = np.all(y > 0) or np.all(y < 0)
        if logy is None:
            logy = same and (np.abs(y).max() / max(np.abs(y).min(), 1e-12) > 100)
        self.logy = bool(logy)
        self.same_sign = bool(same)
        self._S = None
        # train/val 固定切分（在训练集内再切）
        n = len(y)
        rr = np.random.default_rng(seed).permutation(n)
        nt = n * 3 // 4
        self.tr, self.va = rr[:nt], rr[nt:]

    @property
    def S(self):
        if self._S is None:
            self._S = Surface(self.X, self.y, logy=self.logy,
                              epochs=self.epochs)
        return self._S

    def orig_cols(self):
        return [k for k in range(self.X.shape[1])
                if self.col_expr[k][0] == 'var']


class Observation:
    def __init__(self, attr, confidence, col=None, evidence=None, detail=None):
        self.attr = attr
        self.confidence = float(confidence)
        self.col = col
        self.evidence = evidence
        self.detail = detail

    def __repr__(self):
        return "Obs(%s c=%.3f%s%s)" % (
            self.attr, self.confidence,
            (" col=%s" % self.col) if self.col is not None else "",
            " EV" if self.evidence else "")


# ================= 基础测量（模块级，供探针复用） =================
def _eta2(X, y, k, rows=None, nb=10):
    """条件方差 η²（不依赖模型）。"""
    rows = np.arange(len(y)) if rows is None else rows
    xb, yy = X[rows, k], y[rows]
    if np.var(yy) <= 1e-12:
        return 0.0
    qs = np.unique(np.quantile(xb, np.linspace(0, 1, nb + 1)))
    sse = 0.0
    for b in range(len(qs) - 1):
        if b == len(qs) - 2:
            m = (xb >= qs[b]) & (xb <= qs[b + 1])
        else:
            m = (xb >= qs[b]) & (xb < qs[b + 1])
        if m.sum() > 1:
            sse += np.sum((yy[m] - yy[m].mean()) ** 2)
    sst = np.sum((yy - yy.mean()) ** 2)
    return clamp(1 - sse / max(sst, 1e-12))


def _perm_importance(S, X, y, k, rows=None):
    """置换重要性（误差增量比例）。"""
    rows = np.arange(len(y)) if rows is None else rows
    Xs, ys = X[rows], y[rows]
    base_err = np.mean((ys - S.query(Xs)) ** 2)
    Xp = Xs.copy()
    Xp[:, k] = np.random.default_rng(300 + k).permutation(Xp[:, k])
    perm_err = np.mean((ys - S.query(Xp)) ** 2)
    return (perm_err - base_err) / (base_err + 1e-9)


def _sweep_curve(S, X, k, ncurves=10):
    grid = S.grid_i(k)
    idx = np.random.default_rng(100 + k).choice(
        len(X), min(ncurves, len(X)), replace=False)
    curves = [S.sweep(X[b], k, grid) for b in idx]
    return grid, curves


def _sweep_mult_conf(S, X, y, k):
    """同号 sweep：忠实反映 v3 硬判据（me<0.08、斜率干净、整数幂）。
    返回 (conf, power or None, me)。"""
    grid, curves = _sweep_curve(S, X, k)
    me = _mult_err(curves)
    slope = None
    if np.all(curves[0] > 0) or np.all(curves[0] < 0):
        slope = np.polyfit(np.log(grid), np.log(np.abs(curves[0])), 1)[0]
    if slope is None:
        return 0.0, None, me
    clean = SC.clean_ci(slope)
    ds = abs(clean - slope)
    is_int = abs(clean - round(clean)) < 1e-9 and round(clean) != 0
    if is_int and me < 0.08 and ds < 0.1:          # v3 硬判据
        c = 0.90 + 0.10 * (1 - me / 0.08)
    elif is_int and me < 0.15 and ds < 0.2:        # 边缘
        c = 0.55 + 0.35 * (1 - me / 0.15)
    else:                                          # 不成立
        c = 0.25 * (1 - clamp(me / 0.3))
    return clamp(c), clean, me


def _scale_mult_conf(S, X, y, k):
    """跨零条件尺度：std(y) 随 x^q。v3 判据 r2>0.9、ds<0.08、整数幂。
    返回 (conf, power or None)。"""
    xb = X[:, k]
    nb = 8
    qs = np.unique(np.quantile(xb, np.linspace(0, 1, nb + 1)))
    centers, scales = [], []
    for b in range(len(qs) - 1):
        if b == len(qs) - 2:
            m = (xb >= qs[b]) & (xb <= qs[b + 1])
        else:
            m = (xb >= qs[b]) & (xb < qs[b + 1])
        if m.sum() < 5:
            continue
        s = np.std(y[m])
        if s < 1e-12:
            continue
        centers.append(np.median(xb[m])); scales.append(s)
    if len(centers) < 5:
        return 0.0, None
    lc, ls = np.log(centers), np.log(scales)
    q, ii = np.polyfit(lc, ls, 1)
    fit = q * lc + ii
    r2 = 1 - np.sum((ls - fit) ** 2) / max(np.sum((ls - ls.mean()) ** 2), 1e-12)
    clean = SC.clean_ci(q)
    ds = abs(clean - q)
    is_int = abs(clean - round(clean)) < 1e-9 and round(clean) != 0
    if is_int and r2 > 0.9 and ds < 0.08:
        c = 0.90 + 0.10 * r2
    elif is_int and r2 > 0.8 and ds < 0.15:
        c = 0.55 + 0.35 * r2
    else:
        c = 0.30 * r2
    return clamp(c), clean


def _abs_sweep_mult_conf(S, X, y, k):
    """跨零奇幂乘性：|curve| 归一化一致性 + log-log 线性幂次。

    y 变号（如 cos·sin/r³）时，变量 k 若仍是纯乘性奇幂（r^-3），
    单条 sweep 曲线跨零、带符号归一化失效；改用 |curve|：
    - 乘性：|curve_b| = a_b·|f_k|，abs 归一化一致；
    - 加性：曲线是平移关系，abs 归一化不一致；
    - exp：log-log 线性度差（r2 低），被挡住。
    返回 (conf, power or None, err)。"""
    grid, curves = _sweep_curve(S, X, k)
    ac = [np.abs(c) for c in curves]
    ref = len(curves[0]) // 2
    norms = [c / max(c[ref], 1e-9) for c in ac]
    err = 0.0
    for a in range(len(norms)):
        for b in range(a + 1, len(norms)):
            err = max(err, np.max(np.abs(norms[a] - norms[b]))
                      / max(np.max(np.abs(norms[a])), 1e-9))
    med = np.median(np.array(ac), axis=0)
    ok = med > 1e-9
    if ok.sum() < len(grid) * 0.8:
        return 0.0, None, err
    lg, ls = np.log(grid[ok]), np.log(med[ok])
    q, ii = np.polyfit(lg, ls, 1)
    fit = q * lg + ii
    r2 = 1 - np.sum((ls - fit) ** 2) / max(np.sum((ls - ls.mean()) ** 2), 1e-12)
    clean = SC.clean_ci(q)
    ds = abs(clean - q)
    is_int = abs(clean - round(clean)) < 1e-9 and round(clean) != 0
    if is_int and r2 > 0.95 and err < 0.15 and ds < 0.12:
        c = 0.85 + 0.10 * (1 - err / 0.15)
    elif is_int and r2 > 0.9 and err < 0.2 and ds < 0.15:
        c = 0.55 + 0.30 * (1 - err / 0.2)
    else:
        c = 0.30 * (1 - clamp(err / 0.3))
    return clamp(c), clean, err


def _interaction_matrix(S, X, cols, P=600):
    """局部梯度交互矩阵（梯度幅值相关）。"""
    rng = np.random.default_rng(2)
    lo, hi = X.min(0), X.max(0)
    pts = lo + rng.random((P, X.shape[1])) * (hi - lo)
    eps = 0.05 * (hi - lo)
    g = {}
    for i in cols:
        Xp = pts.copy(); Xp[:, i] += eps[i]
        Xm = pts.copy(); Xm[:, i] -= eps[i]
        g[i] = (S.query(Xp) - S.query(Xm)) / (2 * eps[i])
    m = len(cols)
    M = np.zeros((m, m))
    for a in range(m):
        for b in range(m):
            ga, gb = g[cols[a]], g[cols[b]]
            ca, cb = ga - ga.mean(), gb - gb.mean()
            M[a, b] = np.sum(ca * cb) / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-12)
    return M


def _groups_from_M(M, th=0.25):
    """交互矩阵连通分组（强负相关也是交互，用 abs）。"""
    m = M.shape[0]
    adj = np.abs(M) > th
    groups, unseen = [], set(range(m))
    while unseen:
        s = unseen.pop(); comp = [s]; st = [s]
        while st:
            u = st.pop()
            for v in range(m):
                if v in unseen and (adj[u, v] or adj[v, u]):
                    unseen.remove(v); comp.append(v); st.append(v)
        groups.append(comp)
    return groups


# ================= P1 变量相关性 =================
class RelevanceProbe:
    attr = 'relevance'

    @staticmethod
    def observe(st):
        out = []
        for k in range(st.X.shape[1]):
            def c_on(rows):
                eta = _eta2(st.X, st.y, k, rows)
                pi = _perm_importance(st.S, st.X, st.y, k, rows)
                fpi = clamp(pi / 2.0)
                return 0.5 * eta + 0.5 * fpi
            eta_full = _eta2(st.X, st.y, k)
            c = min(c_on(st.tr), c_on(st.va))
            out.append(Observation('relevance', c, col=k,
                                   detail={'eta': eta_full}))
        return out


# ================= P2 乘性外层 =================
class MultProbe:
    attr = 'mult'

    @staticmethod
    def observe(st):
        out = []
        cols = st.orig_cols()
        for k in cols:
            if st.same_sign:
                c, p, me = _sweep_mult_conf(st.S, st.X, st.y, k)
                if c < 0.3:
                    c2, p2 = _scale_mult_conf(st.S, st.X, st.y, k)
                    # 同号时 scale fallback 必须曲线形状一致（me 小），
                    # 防 exp(C/x) 等强非线性被 std 尺度误判成幂次
                    if c2 > c and me < 0.15:
                        c, p = c2, p2
            else:
                c, p = _scale_mult_conf(st.S, st.X, st.y, k)
                # 跨零奇幂乘性（如 cos·sin/r³ 中的 r^-3）：abs sweep 兜底
                ca, pa, _ = _abs_sweep_mult_conf(st.S, st.X, st.y, k)
                if ca > c:
                    c, p = ca, pa
            ev = None
            if c > 0.8 and p is not None and abs(p - round(p)) < 1e-9 and round(p) != 0:
                ev = {'kind': 'mult', 'col': k, 'power': int(round(p))}
            out.append(Observation('mult', c, col=k, evidence=ev,
                                   detail={'power': p}))
        return out


# ================= P3 加性结构 =================
class AddProbe:
    attr = 'add'

    @staticmethod
    def observe(st):
        cols = list(range(st.X.shape[1]))
        if len(cols) < 2:
            return [Observation('add', 0.0, detail={'groups': []})]
        M = _interaction_matrix(st.S, st.X, cols)
        off = np.array([abs(M[a, b]) for a in range(len(cols))
                        for b in range(len(cols)) if a != b])
        c_add = 1 - float(np.mean(off)) if len(off) else 1.0
        groups = _groups_from_M(M)
        return [Observation('add', clamp(c_add),
                            evidence=None,
                            detail={'groups': groups, 'M': M})]


# ================= P4 外层包裹（sqrt/exp） =================
class OuterProbe:
    attr = 'outer'

    @staticmethod
    def _struct_conf(X, z, st):
        """z 上的结构置信度：乘性（任一干净幂次）或加性（交互矩阵）。"""
        ce = list(st.col_expr); nm = list(st.names)
        sub = State(X, z, ce, nm, logy=False)
        cm = 0.0
        for k in sub.orig_cols():
            c, p = _sweep_mult_conf(sub.S, X, z, k)[:2]
            if c > cm:
                cm = c
        if len(sub.orig_cols()) >= 2:
            M = _interaction_matrix(sub.S, X, sub.orig_cols())
            off = np.array([abs(M[a, b]) for a in range(M.shape[0])
                            for b in range(M.shape[0]) if a != b])
            cm = max(cm, 1 - float(np.mean(off)) if len(off) else 1.0)
        return cm

    @staticmethod
    def observe(st):
        out = []
        # y 上的基线结构
        c_y = 0.0
        for k in st.orig_cols():
            c, _ = _sweep_mult_conf(st.S, st.X, st.y, k)[:2]
            c_y = max(c_y, c)
        # sqrt：z = y^2
        if st.same_sign:
            z = st.y ** 2
            c_z = OuterProbe._struct_conf(st.X, z, st)
            c_sqrt = clamp(c_z - c_y)
            ev = {'kind': 'outer', 'wrap': 'sqrt'} if c_sqrt > 0.15 and c_z > 0.85 else None
            out.append(Observation('outer', c_sqrt, evidence=ev,
                                   detail={'wrap': 'sqrt', 'c_z': c_z}))
        # exp：z = log(y)
        if np.all(st.y > 0):
            z = np.log(np.clip(st.y, 1e-9, None))
            c_z = OuterProbe._struct_conf(st.X, z, st)
            c_exp = clamp(c_z - c_y)
            ev = {'kind': 'outer', 'wrap': 'exp'} if c_exp > 0.15 and c_z > 0.85 else None
            out.append(Observation('outer', c_exp, evidence=ev,
                                   detail={'wrap': 'exp', 'c_z': c_z}))
        return out


# ================= P5 对称 =================
class SymProbe:
    attr = 'sym'

    @staticmethod
    def _symmetric(S, X, i, j):
        rng = np.random.default_rng(77)
        idx = rng.choice(len(X), min(200, len(X)), replace=False)
        P = X[idx].copy(); P2 = P.copy()
        P2[:, i], P2[:, j] = P[:, j], P[:, i]
        f1, f2 = S.query(P), S.query(P2)
        rel = np.abs(f1 - f2) / (np.abs(f1) + np.abs(f2) + 1e-9)
        med, p90 = np.median(rel), np.quantile(rel, 0.9)
        if med < 0.005 and p90 < 0.02:              # v3 硬判据
            c = 0.90 + 0.10 * (1 - med / 0.005)
        elif med < 0.01 and p90 < 0.04:             # 边缘
            c = 0.55 + 0.35 * (1 - med / 0.01)
        else:
            c = 0.25 * (1 - clamp(med / 0.05))
        return clamp(c)

    @staticmethod
    def observe(st):
        out = []
        orig = st.orig_cols()
        for a in range(len(orig)):
            for b in range(a + 1, len(orig)):
                i, j = orig[a], orig[b]
                c = SymProbe._symmetric(st.S, st.X, i, j)
                ev = {'kind': 'sym', 'i': i, 'j': j} if c > 0.9 else None
                out.append(Observation('sym', c, evidence=ev,
                                       detail={'pair': (i, j)}))
        return out


# ================= P6 周期 =================
class PeriodicProbe:
    attr = 'periodic'

    @staticmethod
    def observe(st):
        out = []
        for k in st.orig_cols():
            grid, curves = _sweep_curve(st.S, st.X, k, ncurves=4)
            c = curves[0]
            # 振荡：极值点数（峰+谷）。完整周期需 ext>=2（一个峰+一个谷）
            ext = np.sum((c[1:-1] > c[:-2]) & (c[1:-1] > c[2:])) + \
                  np.sum((c[1:-1] < c[:-2]) & (c[1:-1] < c[2:]))
            # sin+cos 拟合
            gn = (grid - grid.mean()) / (grid.std() + 1e-9)
            cn = (c - c.mean()) / (c.std() + 1e-9)
            A = np.column_stack([np.sin(gn), np.cos(gn), np.ones_like(gn)])
            coef, *_ = np.linalg.lstsq(A, cn, rcond=None)
            r2_per = SC.r2(cn, A @ coef)
            osc = ext >= 2          # 收紧：单极值（单调弯曲）不算周期
            c_per = clamp(r2_per) if osc else 0.0
            # 频率由极值点间距粗估，相位由 sin/cos 比
            freq = None
            if osc:
                span = grid.max() - grid.min()
                freq = (ext + 1) / (2 * span) if span > 0 else None
            phase = float(np.arctan2(coef[1], coef[0]))
            ev = {'kind': 'periodic', 'col': k} if c_per > 0.85 else None
            out.append(Observation('periodic', c_per, col=k, evidence=ev,
                                   detail={'ext': int(ext), 'freq': freq,
                                           'phase': phase}))
        return out


# ================= 可插拔注册表 + 扫描 =================
PROBES = [
    RelevanceProbe,
    MultProbe,
    AddProbe,
    OuterProbe,
    SymProbe,
    PeriodicProbe,
]


def scan_all(st, probes=None):
    probes = probes or PROBES
    obs = []
    for P in probes:
        try:
            obs.extend(P.observe(st))
        except Exception as e:
            obs.append(Observation('error:%s' % P.attr, 0.0, detail={'err': str(e)}))
    return obs


def aggregate_observations(obs):
    """Aggregate observations into a structured summary dict."""
    mi = {'relevance': {}, 'mult': {}, 'add': None, 'outer': [],
          'sym': [], 'periodic': {}}
    for o in obs:
        if o.attr == 'relevance':
            mi['relevance'][o.col] = o.confidence
        elif o.attr == 'mult':
            mi['mult'][o.col] = (o.confidence, o.detail)
        elif o.attr == 'add':
            mi['add'] = (o.confidence, o.detail)
        elif o.attr == 'outer':
            mi['outer'].append((o.confidence, o.evidence, o.detail))
        elif o.attr == 'sym':
            mi['sym'].append((o.confidence, o.evidence, o.detail))
        elif o.attr == 'periodic':
            mi['periodic'][o.col] = (o.confidence, o.detail)
    return mi
