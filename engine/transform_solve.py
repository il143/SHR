# -*- coding: utf-8 -*-
"""Unified "assume -> transform -> verify" solver (V7, no per-problem
patches).

Principle: assume -> transform -> verify r2=1 on real data (replacing the
Surface-based "detect -> threshold").

Operations (all deterministic, bounded candidates, no problem name or
answer):
  1. Layered greedy direct solve (small base + pair, angle features);
  2. Merge dimensionality reduction (lightweight pre-screen after building,
     without Surface);
  3. Peel multivariable multiplicative C (locked in log space, partial peel);
  4. Peel additive constants + trig (local regression estimates constants,
     mask peels cos/sin, including squared paths);
  5. Enumerate single-variable powers (verified after peeling);
  6. Enumerate fraction denominators (verified by multiplying back).
Every candidate is hard-verified by "can it rebuild r2=1 after transform +
symbolic gate", so wrong candidates cannot pass.
The exactness check lives outside this module (the fit does not read ground
truth).
"""
import numpy as np
import os
import time
import itertools
from scipy.spatial import cKDTree

import mult_log_solver as MLS
import structure_tools as ST
import structure_guided as SG
from auto_decompose_v5 import _ev_c

GATE = 1 - 1e-7
POWERS = (-4, -3, -2, -1, 1, 2, 3)
ANGLE_KS = (2, 3)
MAX_MERGE_DEPTH = 2
MAX_PEEL_DEPTH = 8
MAX_TERMS = 8
TOP_MERGE = 8
TRIG_MASK = 0.2
KNN = 16
# peel_additive_trig 硬门：剥掉 trig 后必须比不剥"更可解"（quick r2 提升
# 超过该值，或直接 r2=1）。错误结构（trig 在分母/乘性位）不得按加性展开，
# 否则一次生成多个错误分支、深层递归组合爆炸（test_15、I_43_43 打转根因）。
ATRIG_DELTA = 0.1
# combined_solve 全局时间预算（秒）。该求解器定位是"快速短路"，真正由它解出
# 的题（family=unified）最慢约 16.5s；超过预算说明它不擅长（如 test_15、
# I_16_6，正确解在 find_rational），应快速失败把控制权交回 fit 兜底，
# 而不是在深层递归里组合爆炸。
COMBINED_BUDGET = 25.0
_SOLVE_DEADLINE = None
# 官方 max_time 对应的整题全局预算（fit 开头设置），与 combined 的 25s 预算独立。
_GLOBAL_DEADLINE = None


def set_global_budget(seconds):
    """设置整题全局截止（官方 max_time）；None/非正 => 清除。"""
    global _GLOBAL_DEADLINE
    if seconds and seconds > 0:
        _GLOBAL_DEADLINE = time.time() + float(seconds)
    else:
        _GLOBAL_DEADLINE = None


def _deadline_hit():
    """预算是否耗尽：combined 的 25s 预算，或整题全局预算（官方 max_time）。"""
    now = time.time()
    if _SOLVE_DEADLINE is not None and now > _SOLVE_DEADLINE:
        return True
    return _GLOBAL_DEADLINE is not None and now > _GLOBAL_DEADLINE


# ================= 简单常数 =================

def _simple_const(val, tol=0.04):
    """识别简单常数，返回 float 或 None。"""
    cands = [0.0, 1.0, -1.0, 2.0, -2.0, 3.0, 0.5, -0.5, 0.25, -0.25,
             1.0 / 3, 1.0 / 6, 1.0 / 12, 1.0 / 16]
    for c in cands:
        if abs(val - c) < tol:
            return c
    pi = np.pi
    for c in (pi, -pi, 1.0 / pi, 2.0 / pi, 1.0 / (4 * pi), -1.0 / (4 * pi)):
        if abs(val - c) < tol * max(1.0, abs(c)):
            return c
    return None


# ================= 分层 greedy（验证核心） =================

def greedy_on_cols(X, y, cols, max_terms=MAX_TERMS):
    n = len(y)
    lv, ln, seen = [], [], set()
    for v, nd in cols:
        k = SG._node_key(nd)
        if k not in seen and np.all(np.isfinite(v)) and np.std(v) > 1e-12:
            seen.add(k)
            lv.append(np.asarray(v, float))
            ln.append(nd)
    Xm = np.ones((n, 1))
    sel_v, sel_n, chosen = [], [], set()

    def fit_r2():
        cf, *_ = np.linalg.lstsq(Xm, y, rcond=None)
        rr = 1 - np.sum((y - Xm @ cf) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
        return rr, cf

    for _ in range(max_terms):
        st = SG._fwd_step(y, lv, ln, Xm, chosen, cross_v=sel_v, cross_n=sel_n)
        if st is None:
            break
        v, nd = st
        Xm = SG._add_col(sel_v, sel_n, chosen, Xm, v, nd)
        rr, cf = fit_r2()
        if rr >= GATE:
            return rr, SG._linear_expr(cf, sel_n)
    rr, cf = fit_r2()
    return rr, SG._linear_expr(cf, sel_n)


def angle_cols(X):
    d = X.shape[1]
    cols = []
    for i in range(d):
        v = X[:, i]
        lo, hi = float(v.min()), float(v.max())
        if lo > -0.5 and 0.5 < hi < 7.0:
            for k in ANGLE_KS:
                cols.append((np.sin(v / k), ("sin", ("div", ("c", i), ("const", float(k))))))
                cols.append((np.cos(v / k), ("cos", ("div", ("c", i), ("const", float(k))))))
            cols.append((np.sin(2 * v), ("sin", ("mul", ("const", 2.0), ("c", i)))))
            cols.append((np.cos(2 * v), ("cos", ("mul", ("const", 2.0), ("c", i)))))
    return cols


def angle_power_cols(X):
    """对 angle 复合（sin(theta/2)、cos(theta/2)、sin(2theta) 等）再取幂。
    覆盖 1/sin(theta/2)^2 这类"复合的负幂"（test_1 型）。"""
    out = []
    for v, nd in angle_cols(X):
        av = np.abs(v)
        for p in (-3, -2, -1, 2, 3):
            if p < 0:
                if np.min(av) > 1e-6:
                    out.append((av ** p, ("pow", nd, p)))
            else:
                pv = v ** p
                if np.all(np.isfinite(pv)):
                    out.append((pv, ("pow", nd, p)))
    return out


def sqdiff_power_cols(X):
    """添加式平方差 D = X_i^2 - X_j^2（保留原始列），再取负幂。
    覆盖 omega^4/(omega^2-omega_0^2)^2 型（I_32_17）：分子分母共用同一变量，
    替换式归并会丢失分子，故必须"添加"复合列。候选有限（C(d,2)×3）。"""
    d = X.shape[1]
    out = []
    for i in range(d):
        for j in range(i + 1, d):
            D = X[:, i] ** 2 - X[:, j] ** 2
            Dnode = ("add", ("pow", ("c", i), 2),
                     ("mul", ("const", -1.0), ("pow", ("c", j), 2)))
            ad = np.abs(D)
            for p in (-1, -2, -3):
                if p % 2 == 0:
                    if np.min(ad) > 1e-6:
                        out.append((ad ** p, ("pow", Dnode, p)))
                else:
                    if np.min(ad) > 1e-6:
                        out.append((np.sign(D) * ad ** p, ("pow", Dnode, p)))
    return out


def sqdiff_search(X, y):
    """对每个平方差 D = X_i^2-X_j^2 的每个负幂，独立尝试（原始列 + 这一个 D^p）。
    多幂同时入池时，前向 greedy 第一步可能选错幂（D^-3 压过 D^-2），随后陷入
    局部最优；逐个独立尝试可把"第一步选哪个 D、哪个幂"显式枚举，靠 cross 逐步
    构造 D^p·X^k。候选有限（C(d,2)×3），每个列集小、很快。返回 (r2,node) 或 None。"""
    d = X.shape[1]
    base = [(X[:, k], ("c", k)) for k in range(d)]
    for i in range(d):
        for j in range(i + 1, d):
            D = X[:, i] ** 2 - X[:, j] ** 2
            Dnode = ("add", ("pow", ("c", i), 2),
                     ("mul", ("const", -1.0), ("pow", ("c", j), 2)))
            ad = np.abs(D)
            if np.min(ad) <= 1e-6:
                continue
            for p in (-1, -2, -3):
                if p % 2 == 0:
                    pv = ad ** p
                else:
                    pv = np.sign(D) * ad ** p
                cols = base + [(pv, ("pow", Dnode, p))]
                r, n = greedy_on_cols(X, y, cols)
                if r >= GATE and n is not None and SG.symbol_gate(n):
                    return r, n
    return None


def sqdiff_greedy(X, y):
    """small_base + pair + 平方差负幂一起解（保留，可能被高次幂干扰）。"""
    feats = ["x%d" % i for i in range(X.shape[1])]
    cols = (SG.small_base(X, feats) + SG.pair_cols(X)
            + sqdiff_power_cols(X))
    return greedy_on_cols(X, y, cols)


def _is_angle(v):
    lo, hi = float(v.min()), float(v.max())
    return lo > -0.5 and 0.5 < hi < 7.0


def _is_angular(v):
    """角度变量（含差角，范围约 ±2π）。比 _is_angle 宽松，用于枚举兜底；
    误判由"剥后 r2=1"硬门排除。"""
    lo, hi = float(v.min()), float(v.max())
    return (-6.5 < lo < 6.5 and -6.5 < hi < 6.5
            and (hi - lo) > 1.0 and max(abs(lo), abs(hi)) > 0.5)


def layered_greedy(X, y, extra=None):
    feats = ["x%d" % i for i in range(X.shape[1])]
    cols = SG.small_base(X, feats) + SG.pair_cols(X)
    if extra:
        cols = cols + extra
    return greedy_on_cols(X, y, cols)


def angle_greedy(X, y):
    """angle 复合 + 复合幂一起解（test_1 的 1/sin(theta/2)^2）。"""
    feats = ["x%d" % i for i in range(X.shape[1])]
    cols = (SG.small_base(X, feats) + SG.pair_cols(X)
            + angle_cols(X) + angle_power_cols(X))
    return greedy_on_cols(X, y, cols)


def log_only_cols(X):
    """对每个全正（或全负）变量取 log；log 是加性结构（ln(V2/V1)）。
    单独成列，不与 ratio/幂混（那些会用高次泰勒抢选，见 I_44_4 诊断）。"""
    d = X.shape[1]
    cols = []
    for i in range(d):
        v = X[:, i]
        if np.all(v > 1e-9):
            cols.append((np.log(v), ("log", ("c", i))))
        elif np.all(v < -1e-9):
            cols.append((np.log(-v), ("log", ("c", i))))
    return cols


def log_only_greedy(X, y):
    """原始列 + log 列（不含 ratio/幂/pair，避免高次泰勒干扰）；cross 会构造
    乘性 C × log 加性（n·kb·T·ln(V2/V1)，I_44_4 型）。"""
    d = X.shape[1]
    cols = [(X[:, i], ("c", i)) for i in range(d)]
    cols += log_only_cols(X)
    return greedy_on_cols(X, y, cols)


def quick_greedy(X, y):
    """只小 base（轻量，用于预筛），返回 r2。"""
    feats = ["x%d" % i for i in range(X.shape[1])]
    return greedy_on_cols(X, y, SG.small_base(X, feats))[0]


# ================= 归并（降维） =================

def _combo_make(comp, X, i, j):
    cval = ST._combo_value(comp, X, i, j)
    if not np.all(np.isfinite(cval)):
        return None
    keep = [k for k in range(X.shape[1]) if k != i and k != j]
    newX = np.column_stack([X[:, keep], cval])
    return newX, ST._combo_expr(comp, i, j), keep


def _peel_score(X, y):
    """归并价值评估：(quick r2, 剥 multC 后最好 quick r2)。"""
    q = quick_greedy(X, y)
    best = q
    for inner, _fnode in peel_multC(X, y):
        r = quick_greedy(X, inner)
        if r > best:
            best = r
    return q, best


def _meaningful_combos(X, i, j):
    """根据两列的范围/类型，只生成有意义的归并 comp（机理约束，减少盲枚举）。"""
    vi, vj = X[:, i], X[:, j]
    ai, aj = _is_angle(vi), _is_angle(vj)
    both_pos = vi.min() > 0 and vj.min() > 0
    pi, pj = np.ptp(vi), np.ptp(vj)
    similar = min(pi, pj) > 1e-9 and max(pi, pj) / min(pi, pj) < 3
    combos = []
    if ai and aj:
        combos += ["diff", "sum"]
    if both_pos:
        combos += ["ratio", "prod"]
    if similar:
        combos += ["diff", "sum", "sqdiff", "sqsum"]
    if not combos:
        combos = ["prod"]
    return list(dict.fromkeys(combos))


def merge_candidates(X, y):
    """构造归并，按"归并后剥层潜力"筛选（quick r2 或剥 multC 后 r2 改善）。"""
    d = X.shape[1]
    base_q, base_p = _peel_score(X, y)
    scored = []
    for i in range(d):
        for j in range(i + 1, d):
            for comp in _meaningful_combos(X, i, j):
                m = _combo_make(comp, X, i, j)
                if m is None:
                    continue
                newX, enode, keep = m
                q, p = _peel_score(newX, y)
                if (q >= GATE or p >= GATE
                        or q > base_q + 0.01 or p > base_p + 0.01):
                    scored.append((max(q, p), newX, enode, keep))
    scored.sort(key=lambda t: -t[0])
    return [(newX, enode, keep) for _, newX, enode, keep in scored[:TOP_MERGE]]


def remap_merge(node, enode, keep):
    nkeep = len(keep)

    def conv(n):
        if not isinstance(n, tuple):
            return n
        if n[0] in ("c", "var"):
            k = n[1]
            return ("c", keep[k]) if k < nkeep else enode
        return tuple([n[0]] + [conv(a) for a in n[1:]])

    return conv(node)


# ================= 剥乘性 =================

def _power_node(i, p):
    v = ("c", i)
    if p == 1:
        return v
    if p > 1:
        return ("pow", v, p)
    if p == -1:
        return ("div", ("const", 1.0), v)
    return ("div", ("const", 1.0), ("pow", v, -p))


def peel_multC(X, y):
    """锁多变量乘性因子。log 回归系数就近取整；阈值 0.1 区分真乘性(<0.1)
    与复合/非乘性污染(omega/omega_0 等偏差>0.1)；幂次 0 不锁；剥后 r2 验证。"""
    d = X.shape[1]
    ok = np.abs(y) > 1e-9
    for j in range(d):
        ok &= np.abs(X[:, j]) > 1e-9
    if ok.sum() < d + 2:
        return []
    Xo, yo = X[ok], y[ok]
    A = np.column_stack([np.ones(len(yo)), np.log(np.abs(Xo))])
    coef = np.linalg.lstsq(A, np.log(np.abs(yo)), rcond=None)[0]
    k = coef[1:]
    ki = [int(round(v)) for v in k]
    # 干净阈值 0.15：归一化数据上内层加性会轻微污染 log 系数（如 test_20
    # 的 omega_0=1.865 真值 2），0.1 过严会漏锁；放宽到 0.15，误锁由
    # 下方"剥后 r2 不劣化"硬门拒绝。
    clean = [(j, ki[j]) for j in range(d)
             if ki[j] != 0 and abs(k[j] - ki[j]) < 0.15]
    if not clean:
        return []
    factor = np.ones(len(y))
    for j, p in clean:
        ax = np.abs(X[:, j])
        factor *= ax ** p if p % 2 == 0 else np.sign(X[:, j]) * ax ** p
    fnode = _power_node(clean[0][0], clean[0][1])
    for j, p in clean[1:]:
        fnode = ("mul", fnode, _power_node(j, p))
    inner = y / factor
    if not np.all(np.isfinite(inner)):
        return []
    # 剥后 r2 必须不劣于原始（误锁会使 r2 变差）
    if quick_greedy(X, inner) + 0.005 < quick_greedy(X, y):
        return []
    return [(inner, fnode)]


def peel_single_power(X, y):
    d = X.shape[1]
    base = quick_greedy(X, y)
    out = []
    for i in range(d):
        v = X[:, i]
        for p in POWERS:
            if p < 0 and np.min(np.abs(v)) < 1e-6:
                continue
            f = np.abs(v) ** p
            if not np.all(np.isfinite(f)) or np.min(np.abs(f)) < 1e-9:
                continue
            inner = y / f
            if not np.all(np.isfinite(inner)):
                continue
            r = quick_greedy(X, inner)
            if r >= GATE or r > base + 0.01:
                out.append((r, inner, _power_node(i, p)))
    out.sort(key=lambda t: -t[0])
    return [(inner, fn) for _, inner, fn in out]


# ================= 剥加性常数 + 三角 =================

def peel_additive_trig(X, y):
    """对角度变量局部回归估加性常数 c，剥 c 后 mask 剥 trig（含平方）。
    返回 [(subX, suby, 逆变换函数)]。
    """
    d = X.shape[1]
    out = []
    for ang in range(d):
        if _deadline_hit():
            break
        if not _is_angle(X[:, ang]):
            continue
        other = [k for k in range(d) if k != ang]
        if not other:
            continue
        Z = X[:, other]
        sc = np.ptp(Z, axis=0)
        sc[sc < 1e-12] = 1.0
        Zn = Z / sc
        try:
            dd_all, nb_all = cKDTree(Zn).query(Zn, k=min(KNN, len(y)))
        except Exception:
            continue
        for trig, tf in (("cos", np.cos), ("sin", np.sin)):
            t = tf(X[:, ang])
            c_est = np.full(len(y), np.nan)
            for a in range(len(y)):
                g = nb_all[a]
                if len(g) < 6:
                    continue
                Am = np.column_stack([np.ones(len(g)), t[g]])
                cf, *_ = np.linalg.lstsq(Am, y[g], rcond=None)
                c_est[a] = cf[0]
            okm = np.isfinite(c_est)
            if okm.sum() < 20:
                continue
            c = _simple_const(float(np.median(c_est[okm])))
            if c is None:
                continue
            w = y - c
            mask = np.abs(t) > TRIG_MASK
            if mask.sum() < 30:
                continue
            subX = X[np.ix_(mask, other)]
            tnode = ("c", ang)
            wm = w[mask]; tm = t[mask]
            # 非平方：y = c + inner * t
            y1 = wm / tm
            if np.all(np.isfinite(y1)):
                # 硬门：剥掉 t 后必须比不剥更可解（同 mask、同 subX）。
                # 否则错误结构（t 在分母/乘性位）也被按加性展开，制造大量
                # 无效分支（test_15、I_43_43 打转根因）。
                base1 = quick_greedy(subX, wm)
                r1 = quick_greedy(subX, y1)
                if r1 >= GATE or r1 > base1 + ATRIG_DELTA:
                    def inv1(inner, c=c, tnode=tnode, trig=trig):
                        return ("add", ("const", c),
                                ("mul", inner, (trig, tnode)))
                    out.append((subX, y1, inv1))
            # 平方：y = c + sqrt(inner) * t
            y2 = wm ** 2 / tm ** 2
            if np.all(np.isfinite(y2)):
                base2 = quick_greedy(subX, wm ** 2)
                r2 = quick_greedy(subX, y2)
                if r2 >= GATE or r2 > base2 + ATRIG_DELTA:
                    def inv2(inner, c=c, tnode=tnode, trig=trig):
                        return ("add", ("const", c),
                                ("mul", ("sqrt", inner), (trig, tnode)))
                    out.append((subX, y2, inv2))
    return out


# ================= 枚举乘性外层 + 加性常数 + 平方路径 =================

def _local_intercept(X, y, other, t, K=16):
    """在 other 列（非角度）空间 KNN，局部回归 y=a+b*t，返回逐点截距 a。"""
    Z = X[:, other]
    sc = np.ptp(Z, axis=0)
    sc[sc < 1e-12] = 1.0
    Zn = Z / sc
    try:
        _, nb = cKDTree(Zn).query(Zn, k=min(K + 1, len(y)))
    except Exception:
        return None
    a_est = np.full(len(y), np.nan)
    for i in range(len(y)):
        g = nb[i]
        if len(g) < 6:
            continue
        Am = np.column_stack([np.ones(len(g)), t[g]])
        cf = np.linalg.lstsq(Am, y[g], rcond=None)[0]
        a_est[i] = cf[0]
    return a_est, nb


def peel_mult_enum(X, y):
    """枚举"乘性外层 C × (c0 + sqrt(内层)·trig(ang))"。
    当 peel_multC 被加性内层污染（test_2 型强耦合）时兜底：
      局部回归截距给出 C 的参与变量方向 → 枚举幂次 → 剥后截距≈简单常数
      → 平方路径 → solve_mult_log 锁内层。
    返回完整节点（r2=1）或 None。
    """
    d = X.shape[1]
    angle_idx = [i for i in range(d) if _is_angular(X[:, i])]
    if not angle_idx:
        return None
    for ang in angle_idx:
        other = [k for k in range(d) if k != ang]
        m = len(other)
        Z = X[:, other]
        for trig, tf in (("cos", np.cos), ("sin", np.sin)):
            t = tf(X[:, ang])
            li = _local_intercept(X, y, other, t)
            if li is None:
                continue
            a_est, nb = li
            okm = np.isfinite(a_est)
            if okm.sum() < 20:
                continue
            # a_est log 系数 → 参与变量（|coef|>0.3）
            ok = np.abs(a_est) > 1e-6
            for jj in range(m):
                ok &= np.abs(Z[:, jj]) > 1e-9
            if ok.sum() < m + 2:
                continue
            A2 = np.column_stack([np.ones(ok.sum()), np.log(np.abs(Z[ok]))])
            cf = np.linalg.lstsq(A2, np.log(np.abs(a_est[ok])), rcond=None)[0]
            coef = cf[1:]
            part = [jj for jj in range(m) if abs(coef[jj]) > 0.3]
            if not part or len(part) > 4:
                continue
            if os.environ.get("TS_DEBUG"):
                print("    [dbg] ang=%d trig=%s coef=%s part=%s" %
                      (ang, trig, np.round(coef, 2), part))
            # 枚举参与变量幂次
            for pows in itertools.product((-2, -1, 1, 2), repeat=len(part)):
                factor = np.ones(len(y))
                fparts = []
                for jj, p in zip(part, pows):
                    col = other[jj]
                    v = X[:, col]
                    factor *= (np.abs(v) ** p if p % 2 == 0
                               else np.sign(v) * np.abs(v) ** p)
                    fparts.append(_power_node(col, p))
                z = y / factor
                if not np.all(np.isfinite(z)):
                    continue
                # 剥后局部回归截距
                a2 = np.full(len(z), np.nan)
                for i in range(len(z)):
                    g = nb[i]
                    if len(g) < 6:
                        continue
                    Am = np.column_stack([np.ones(len(g)), t[g]])
                    cf2 = np.linalg.lstsq(Am, z[g], rcond=None)[0]
                    a2[i] = cf2[0]
                ok2 = np.isfinite(a2)
                if ok2.sum() < 20:
                    continue
                c0 = _simple_const(float(np.median(a2[ok2])))
                if c0 is None:
                    continue
                w = z - c0
                mask = np.abs(t) > 0.3
                if mask.sum() < 30:
                    continue
                # 平方路径：z = c0 + sqrt(B)·t，B = w²/t²。
                # 注意 solve_mult_log 返回的 expr 已含常数 C（mul(const C,...)）。
                # 先试 B 纯乘性；再试 B = 1 + A（A 纯乘性，减 1 后锁 A）。
                B_raw = w[mask] ** 2 / t[mask] ** 2
                Xsub = X[np.ix_(mask, other)]
                ce = [("c", jj) for jj in range(m)]
                Bnode = None
                if np.all(B_raw > 0):
                    msol = MLS.solve_mult_log(Xsub, B_raw, ce)
                    if msol is not None:
                        Bnode = msol[1]   # 完整 B（含常数）
                if Bnode is None:
                    A_est = B_raw - 1.0
                    if np.all(A_est > 0):
                        msol = MLS.solve_mult_log(Xsub, A_est, ce)
                        if msol is not None:
                            Bnode = ("add", ("const", 1.0), msol[1])
                if Bnode is None:
                    continue

                def remapA(n):
                    if not isinstance(n, tuple):
                        return n
                    if n[0] in ("c", "var"):
                        return ("c", other[n[1]])
                    return tuple([n[0]] + [remapA(a) for a in n[1:]])

                innerB = remapA(Bnode)
                fnode = fparts[0]
                for fp in fparts[1:]:
                    fnode = ("mul", fnode, fp)
                whole = ("mul", fnode,
                         ("add", ("const", float(c0)),
                          ("mul", ("sqrt", innerB), (trig, ("c", ang)))))
                if os.environ.get("TS_DEBUG"):
                    print("    [dbg]   pows=%s c0=%s node_r2=%.6f gate=%s" %
                          (pows, c0, _node_r2(whole, X, y),
                           SG.symbol_gate(whole)))
                if _accept(whole, X, y):
                    return whole
    return None


# ================= 分式分母 =================

def _denom_1plus_cands(X, y):
    """y = 1/(1±u) 型分母的通用反演识别。
    反演 g=1/y，取残差 (g-1) 或 (1-g)，识别残差能否被单一简单组合表示
    （单变量 / 两变量积 / 两变量比 / 两变量差，带一个简单常数缩放）。
    覆盖 1+chi（残差=chi）、1-n·alpha/3（残差=n·alpha/3）等；
    不枚举"1±所有组合"，而是反演后识别，候选少。误判由"残差 r2=1、
    系数简单、最终 whole 过 symbol_gate"硬门排除。返回 [(z=常数1, gnode)]。"""
    d = X.shape[1]
    if not np.all(np.isfinite(y)) or np.min(np.abs(y)) < 1e-4:
        return []
    g = 1.0 / y
    combos = []
    for i in range(d):
        combos.append((X[:, i], ("c", i)))
    for i in range(d):
        for j in range(i + 1, d):
            combos.append((X[:, i] * X[:, j],
                           ("mul", ("c", i), ("c", j))))
            if np.min(np.abs(X[:, j])) > 1e-6:
                combos.append((X[:, i] / np.clip(X[:, j], 1e-9, None),
                               ("div", ("c", i), ("c", j))))
            if np.min(np.abs(X[:, i])) > 1e-6:
                combos.append((X[:, j] / np.clip(X[:, i], 1e-9, None),
                               ("div", ("c", j), ("c", i))))
            combos.append((X[:, i] - X[:, j],
                           ("sub", ("c", i), ("c", j))))
    out = []
    for sgn, v in ((1, g - 1.0), (-1, 1.0 - g)):
        # 残差须有变化且不围绕 0（u 是单一组合，不围绕 0）
        if np.max(np.abs(v)) < 1e-6:
            continue
        if not (np.all(v >= -1e-7) or np.all(v <= 1e-7)):
            continue
        ss = np.sum((v - v.mean()) ** 2)
        for uv, un in combos:
            if not np.all(np.isfinite(uv)):
                continue
            denom = np.sum(uv * uv)
            if denom < 1e-12:
                continue
            c = np.sum(uv * v) / denom
            if not np.isfinite(c) or abs(c) < 1e-9:
                continue
            r = 1 - np.sum((v - c * uv) ** 2) / max(ss, 1e-12)
            if r >= GATE and SG._coeff_simple(c):
                cnode = un if abs(c - 1) < 1e-9 else (
                    "mul", ("const", float(c)), un)
                if sgn == 1:
                    gnode = ("add", ("const", 1.0), cnode)
                else:
                    gnode = ("sub", ("const", 1.0), cnode)
                out.append((np.ones(len(y)), gnode))
                break
    return out


def peel_denom(X, y):
    d = X.shape[1]
    gx = []
    for i in range(d):
        gx.append((X[:, i], ("c", i)))
        # 常数偏移分母（1+X_i / 1-X_i）：物理公式常见（1+chi、1-x 等），
        # 与 rational_blocks.build_denom_factors 的 A 部分对齐；误判由
        # "y*g 能被干净解 r2=1 且 symbol_gate" 硬门排除。
        for c in (1.0, -1.0):
            gx.append((X[:, i] + c, ("add", ("c", i), ("const", c))))
    for i in range(d):
        for j in range(i + 1, d):
            gx.append((X[:, i] + X[:, j], ("add", ("c", i), ("c", j))))
            gx.append((X[:, i] * X[:, j], ("mul", ("c", i), ("c", j))))
    out = []
    for gv, gn in gx:
        if _deadline_hit():
            break
        z = y * gv
        if np.all(np.isfinite(z)):
            r = quick_greedy(X, z)
            if r >= GATE:
                out.append((r, z, gn))
    # 1±u 反演识别（1+chi、1-n·alpha/3 等）
    for z, gn in _denom_1plus_cands(X, y):
        out.append((1.0, z, gn))
    out.sort(key=lambda t: -t[0])
    return [(z, gn) for _, z, gn in out]


def _pmc_factor_cands(X, y):
    """peel_mult_core 的宽松乘性因子：单变量幂（log 系数就近，偏差<.4）
    + 单角 sin/cos。比 peel_multC（.15）宽松，用于剥层后 core 仍解不出的
    深层题（test_14 型）；误判由"剥后 core 能 r2=1 且 symbol_gate"拒绝。"""
    d = X.shape[1]
    out = []
    ok = np.abs(y) > 1e-9
    for j in range(d):
        ok &= np.abs(X[:, j]) > 1e-9
    if ok.sum() >= d + 2:
        A = np.column_stack([np.ones(int(ok.sum())),
                             np.log(np.abs(X[ok]))])
        k = np.linalg.lstsq(A, np.log(np.abs(y[ok])), rcond=None)[0][1:]
        for j in range(d):
            kj = int(round(k[j]))
            if kj != 0 and abs(k[j] - kj) < 0.4:
                ax = np.abs(X[:, j])
                f = ax ** kj if kj % 2 == 0 else np.sign(X[:, j]) * ax ** kj
                if np.all(np.isfinite(f)) and np.min(np.abs(f)) > 1e-9:
                    out.append((f, _power_node(j, kj)))
    for val, nd, ai in _angle_factor_cands(X):
        if nd[0] in ("sin", "cos") and nd[1] == ("c", ai):
            if np.min(np.abs(val)) > 1e-3:
                out.append((val, nd))
    return out


def peel_mult_core(X, y, depth=0, seen=None):
    """剥乘性/角度因子 + super_pool core 验证（test_14 型）。
    先 core 直接解；再枚举宽松因子（偏差<.4、单角），对每个因子算剥后
    core r2（单次 core 约 .15s）：剥后 core 直接 r2=1 即返回；否则按 r2
    排序，只对最有希望的前 2 个递归，depth<4。避免在错误因子上盲目 DFS。
    seen 防重复因子。返回节点或 None。"""
    if seen is None:
        seen = set()
    feats = ["x%d" % i for i in range(X.shape[1])]
    pool = SG.super_pool(X, feats)
    r2, nd = greedy_on_cols(X, y, pool)
    if r2 >= GATE and nd is not None and SG.symbol_gate(nd):
        return nd
    if depth >= 4:
        return None
    scored = []
    for f, fnode in _pmc_factor_cands(X, y):
        key = SG._node_key(fnode)
        if key in seen:
            continue
        inner = y / f
        if not np.all(np.isfinite(inner)):
            continue
        r2i, ni = greedy_on_cols(X, inner, pool)
        scored.append((r2i, f, fnode, ni, inner, key))
    # 剥后 core 直接成功
    for r2i, f, fnode, ni, inner, key in sorted(scored, key=lambda t: -t[0]):
        if r2i >= GATE and ni is not None and SG.symbol_gate(ni):
            return ("mul", fnode, ni)
    # 按剥后 r2 排序，只递归最有希望的前 2 个（且剥后确有改善）
    for r2i, f, fnode, ni, inner, key in sorted(scored, key=lambda t: -t[0])[:2]:
        if r2i <= r2 + 0.02:
            continue
        r = peel_mult_core(X, inner, depth + 1, seen | {key})
        if r is not None:
            return ("mul", fnode, r)
    return None


# ================= 主求解（递归） =================

def _node_r2(node, X, y):
    """节点在当前 X/y 上的数值 r2。"""
    try:
        pred = _ev_c(node, X)
    except Exception:
        return -1.0
    if not np.all(np.isfinite(pred)):
        return -1.0
    ss = np.sum((y - y.mean()) ** 2)
    return 1 - np.sum((y - pred) ** 2) / max(ss, 1e-12)


def _accept(node, X, y):
    """符号合法 + 数值 r2 达标才接受。"""
    return node is not None and SG.symbol_gate(node) and _node_r2(node, X, y) >= GATE


def peel_outer_sqrt(X, y):
    """y = B^2（test_1 型）：B=sqrt(|y|)，有界检查 B 能否被干净解。
    判据：B 直接（或剥乘性后）angle_greedy 能 r2=1 且符号合法；
    全正但非平方（test_20）剥后 r2 明显 < GATE，自动排除。"""
    if np.min(y) > 0:
        B = np.sqrt(y)
    elif np.max(y) < 0:
        B = -np.sqrt(-y)
    else:
        return None
    r, n = angle_greedy(X, B)
    if r >= GATE and n is not None and SG.symbol_gate(n):
        whole = ("pow", n, 2)
        if _accept(whole, X, y):
            return whole
    for inner, fnode in peel_multC(X, B):
        r2, n2 = angle_greedy(X, inner)
        if r2 >= GATE and n2 is not None and SG.symbol_gate(n2):
            whole = ("pow", ("mul", fnode, n2), 2)
            if _accept(whole, X, y):
                return whole
    return None


def _angle_factor_cands(X):
    """枚举角度因子（单/倍/半角 + sin·cos 乘积），返回 (值, 节点, 来源变量)。
    所有变量都可能被归一化到相近范围，故不预先区分角度变量。"""
    d = X.shape[1]
    facs = []
    for ai in range(d):
        v = X[:, ai]
        vn = ("c", ai)
        items = [
            ("sin", ("sin", vn)),
            ("cos", ("cos", vn)),
            ("sin2", ("sin", ("mul", ("const", 2.0), vn))),
            ("cos2", ("cos", ("mul", ("const", 2.0), vn))),
            ("sinhalf", ("sin", ("div", vn, ("const", 2.0)))),
            ("coshalf", ("cos", ("div", vn, ("const", 2.0)))),
            ("sincos", ("mul", ("sin", vn), ("cos", vn))),
        ]
        for nm, nd in items:
            try:
                val = _ev_c(nd, X)
            except Exception:
                continue
            if np.all(np.isfinite(val)):
                facs.append((val, nd, ai))
    return facs


def peel_angle_factor(X, y):
    """角度因子污染乘性识别（II_6 型：cos·sin 使 log 系数全不干净）。
    两阶段：先用 log 回归快速筛"因子来源变量系数≈0"（src_ok）；再对这些
    因子剥后用 super_pool core 真算验证——剥后 core r2>=GATE 才接受。
    只用 log 系数判据在"加性+乘性混合"时不可靠（会把错误因子判干净、
    正确因子判不干净），core 真算兜底。返回 (rest, nd) 列表。"""
    d = X.shape[1]
    feats = ["x%d" % i for i in range(d)]
    pool = SG.super_pool(X, feats)
    out = []
    for val, nd, ai in _angle_factor_cands(X):
        # log 回归用 mask（排除因子接近零的点）
        m = np.abs(val) > 1e-3
        m &= np.abs(y) > 1e-9
        for j in range(d):
            m &= np.abs(X[:, j]) > 1e-9
        if m.sum() < d + 2:
            continue
        rest = y / val
        if not np.all(np.isfinite(rest)):
            continue
        A = np.column_stack([np.ones(m.sum()), np.log(np.abs(X[m]))])
        k = np.linalg.lstsq(A, np.log(np.abs(rest[m])), rcond=None)[0][1:]
        # 第一阶段：因子来源变量系数≈0
        if abs(k[ai]) > 0.1:
            continue
        # 第二阶段：剥后 super_pool core 真算 r2=1
        r2c, nc = greedy_on_cols(X, rest, pool)
        if r2c >= GATE and nc is not None and SG.symbol_gate(nc):
            out.append((rest, nd))
    return out


def transform_solve(X, y, depth=0, allow_merge=True, merged=False,
                    allow_single=True):
    if depth > MAX_PEEL_DEPTH:
        return None
    if _deadline_hit():
        return None
    # 1. 直接解（普通 + 角度特征 + 复合幂）
    r2, node = layered_greedy(X, y)
    if r2 >= GATE and node is not None and SG.symbol_gate(node):
        return node
    r2a, nodea = angle_greedy(X, y)
    if r2a >= GATE and nodea is not None and SG.symbol_gate(nodea):
        return nodea
    # 1c. 纯 log 加性（乘性 C × ln，I_44_4 型；不含 ratio/幂，避免高次
    # 泰勒抢选）。判据严格（r2=1 且 gate），不会误判。
    r2l, nodel = log_only_greedy(X, y)
    if r2l >= GATE and nodel is not None and SG.symbol_gate(nodel):
        return nodel
    # 1b. 整个 y 是某表达式的平方（y = B^2，test_1 型，有界检查）
    if depth < 2:
        node_sq = peel_outer_sqrt(X, y)
        if node_sq is not None:
            return node_sq
    # 2. 剥多变量乘性 C（最常见外层，先做）
    for inner, fnode in peel_multC(X, y):
        r = transform_solve(X, inner, depth + 1,
                            allow_merge=allow_merge, merged=merged,
                            allow_single=allow_single)
        if r is not None:
            whole = ("mul", fnode, r)
            if _accept(whole, X, y):
                return whole
    # 2a. 1±u 反演分母（1+chi、1-n·alpha/3 等）：便宜（~ms）、判据严格
    # （反演残差 r2=1、系数简单），提前到 angle_factor/merge/single_power 等
    # 大量分支之前，避免在错误分支的深层递归里打转。
    for _z, gn in _denom_1plus_cands(X, y):
        whole = ("div", ("const", 1.0), gn)
        if _accept(whole, X, y):
            return whole
    # 2c. 平方差负幂（I_32_17 型：添式复合，分子分母共用变量）。
    # 判据严格（r2=1 且 gate），不会误判；提前到 angle/merge/denom 之前，
    # 避免在这些分支的深层递归里耗时；只在浅层跑，控制开销。
    if depth < 3:
        hit = sqdiff_search(X, y)
        if hit is not None and _accept(hit[1], X, y):
            return hit[1]
    # 2b. 角度因子污染乘性识别（II_6 型）：剥角度因子后递归
    if depth < 4:
        for inner, fnode in peel_angle_factor(X, y):
            r = transform_solve(X, inner, depth + 1,
                                allow_merge=allow_merge, merged=merged,
                                allow_single=allow_single)
            if r is not None:
                whole = ("mul", fnode, r)
                if _accept(whole, X, y):
                    return whole
    # 2d. 宽松剥乘性/角度 + super_pool core（test_14 型：乘性因子与加性内层
    # 混合、某变量同时出现在加性和乘性，严格 log 判据剥不出）。只在顶层跑
    # 一次（它自身是完整剥层 DFS）；对非乘性题 1-5s 快速返回 None，不打转。
    if depth == 0:
        node_pmc = peel_mult_core(X, y)
        if node_pmc is not None and _accept(node_pmc, X, y):
            return node_pmc
    # 3. 归并后优先 peel_mult_enum（强耦合，test_2 型，角度已归并）
    if merged and depth < 4:
        node_enum = peel_mult_enum(X, y)
        if node_enum is not None:
            return node_enum
    # 4. 归并降维（差角/比值等）
    if allow_merge and depth < MAX_MERGE_DEPTH:
        for newX, enode, keep in merge_candidates(X, y):
            inner = transform_solve(newX, y, depth + 1,
                                    allow_merge=True, merged=True)
            if inner is not None:
                back = remap_merge(inner, enode, keep)
                if _accept(back, X, y):
                    return back
    # 4b. 未归并空间兜底 peel_mult_enum
    if not merged and depth < 3:
        node_enum = peel_mult_enum(X, y)
        if node_enum is not None:
            return node_enum
    # 5. 枚举单变量幂次（剥一次后不再剥，多变量幂次由 peel_multC 锁）
    if allow_single:
        for inner, fnode in peel_single_power(X, y):
            r = transform_solve(X, inner, depth + 1,
                                allow_merge=False, merged=merged,
                                allow_single=False)
            if r is not None:
                whole = ("mul", fnode, r)
                if _accept(whole, X, y):
                    return whole
    # 6. 分式分母（含加性 m1+m2，I_18 型；先于三角枚举，避免无三角时误判）
    for z, gn in peel_denom(X, y):
        r2z, nz = layered_greedy(X, z)
        if r2z >= GATE and nz is not None and SG.symbol_gate(nz):
            whole = ("div", nz, gn)
            if _accept(whole, X, y):
                return whole
    # 6b. 平方差负幂（I_32_17 型：分子分母共用变量，添式复合）
    # 先逐个独立枚举（避免多幂入池 greedy 选错幂），再退回合并池
    hit = sqdiff_search(X, y)
    if hit is not None:
        ns = hit[1]
        if _accept(ns, X, y):
            return ns
    r2s, ns = sqdiff_greedy(X, y)
    if r2s >= GATE and ns is not None and SG.symbol_gate(ns):
        if _accept(ns, X, y):
            return ns
    # 7. 加性常数 + 三角（较少见，后置）
    for subX, suby, inv in peel_additive_trig(X, y):
        inner = transform_solve(subX, suby, depth + 1,
                                allow_merge=True, merged=merged,
                                allow_single=allow_single)
        if inner is not None:
            whole = inv(inner)
            if _accept(whole, X, y):
                return whole
    return None


def unified_solve(X, y):
    """对外入口：返回 (r2, node) 或 None；节点变量统一为 var。"""
    node = transform_solve(X, y)
    if node is None:
        return None
    node = SG.c2var(node)
    pred = _ev_c(node, X)
    r2 = 1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
    if r2 >= GATE:
        return r2, node
    return None


def combined_solve(X, y, feats):
    """组合求解：先 SHRv5 全列 greedy（pair/triple/三角/复合角常数，覆盖
    III_8_54 这类"多变量乘性角 + 2pi"）；不达标再走剥层 unified_solve（覆盖
    乘性外层/角度因子/分母/外层 sqrt 等）。两套列空间互补。
    返回 (r2, node) 或 None。"""
    import auto_decompose_v5 as V5
    global _SOLVE_DEADLINE
    _SOLVE_DEADLINE = time.time() + COMBINED_BUDGET
    try:
        col_expr = [('var', i) for i in range(X.shape[1])]
        b = V5.SHRv5(X, y, feats)
        X2, cex2, ce2, nm2, deps2 = b._build_all(X, y, col_expr)
        cols = [(X2[:, j], cex2[j]) for j in range(X2.shape[1])]
        r2, node = greedy_on_cols(X, y, cols)
        if r2 >= GATE:
            return r2, node
        return unified_solve(X, y)
    finally:
        _SOLVE_DEADLINE = None
