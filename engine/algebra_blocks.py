# -*- coding: utf-8 -*-
"""J4 A 代数复合反演（玻尔兹曼 / 双曲外层）。

结构：y = Cn * Nstruct * core(u)

  core ∈ { exp(u)-1,  1/(exp(u)-1),  1/(exp(u)+exp(-u)) [=1/(2·cosh u)],  tanh(u) }
  Nstruct：列的简单乘性组合（常数 / 单列 / pair 积 / 双向比 / 倒数），d<=5
  u：单项式分数（C·∏ xj^kj），用 MLS 解

机理：
  1) 枚举 Nstruct，归一化 z = y/Nstruct；
  2) 对每个 core，取反演“可靠点”——反演函数在饱和区 / 奇点附近对数值误差极敏感
     （tanh 饱和、1/(exp-1) 在 u→0 时 y→∞、1/(2cosh) 在 u→0 时分母→2），
     剔除这些点、用可靠点定结构；这是数值可靠性处理（非凑结果），最后全量
     （含饱和点）严格后验；
  3) 外层常数 Cn：连续最小化“可靠点 log 回归残差 + 幂次偏离整数”这一连续目标，
     避免对阶跃目标（MLS 严格整数门）做连续优化导致不收敛；
  4) 反演 u，MLS 严格整数幂验证；全量严格后验 r2=1。
  5) 表达式用 exp 组装（tanh / cosh 全部用 exp 表达），不依赖 _ev_c 的 tanh 分支。

探针是评估，真值靠全量后验；无逐题分支、无真值依赖。
"""
import numpy as np
from itertools import combinations, permutations
from scipy.optimize import minimize_scalar

import mult_log_solver as MLS
import shr_collapse as SC
from auto_decompose_v5 import _ev_c

ALG_MAX_D = 5
_EPS = 1e-9
SOLVE_R2 = 1.0 - 1e-7
_MIN_REL = 25          # 可靠点最少
_CORES = ["exp-1", "1/(exp-1)", "1/(2cosh)", "tanh"]


# ============================================================
# 归一化结构 Nstruct
# ============================================================
def _norm_structs(X, ce):
    d = X.shape[1]
    out = [(np.ones(len(X)), ("const", 1.0))]
    for i in range(d):
        out.append((X[:, i], ce[i]))
        out.append((1.0 / np.clip(X[:, i], 1e-9, None),
                    ("div", ("const", 1.0), ce[i])))
    for i, j in combinations(range(d), 2):
        out.append((X[:, i] * X[:, j], ("mul", ce[i], ce[j])))
    for i, j in permutations(range(d), 2):
        out.append((X[:, i] / np.clip(X[:, j], 1e-9, None),
                    ("div", ce[i], ce[j])))
    return out


def _reliable(core, z):
    """可靠点布尔：反演函数灵敏区（保证有效数字）。z=core(u) 的 Cn=1 归一化。"""
    if core == "exp-1":
        return (1.0 + z) > 0.1
    if core == "1/(exp-1)":
        return (z > 0) & (z < 20.0)
    if core == "1/(2cosh)":
        return (z > 0) & (1.0 / (2.0 * np.clip(z, 1e-9, None)) > 1.02)
    if core == "tanh":
        return np.abs(z) < 0.9
    return np.zeros(len(z), bool)


def _invert(core, y, nval, Cn):
    """反演得 u（y = Cn*nval*core(u)）。数值非法返回 None。"""
    if core == "exp-1":
        a = 1.0 + y / (Cn * nval)
        if not np.all(a > 0):
            return None
        return np.log(a)
    if core == "1/(exp-1)":
        if not np.all(y > 0):
            return None
        a = 1.0 + Cn * nval / y
        if not np.all(a > 0):
            return None
        return np.log(a)
    if core == "1/(2cosh)":
        if not np.all(y > 0):
            return None
        a = Cn * nval / (2.0 * y)
        if not np.all(a >= 1.0):
            return None
        return np.arccosh(a)
    if core == "tanh":
        a = np.clip(y / (Cn * nval), -1.0 + 1e-8, 1.0 - 1e-8)
        return np.arctanh(a)


def _core_val(core, u):
    if core == "exp-1":
        return np.exp(u) - 1.0
    if core == "1/(exp-1)":
        return 1.0 / (np.exp(u) - 1.0)
    if core == "1/(2cosh)":
        return 1.0 / (np.exp(u) + np.exp(-u))
    if core == "tanh":
        return np.tanh(u)


def _logreg_cont(X, u):
    """连续 log 回归（不取整）：返回 (r2, 连续幂次 k)。"""
    A = np.column_stack([np.ones(len(u)), np.log(np.abs(X))])
    lu = np.log(np.abs(u))
    coef = np.linalg.lstsq(A, lu, rcond=None)[0]
    pred = A @ coef
    ss = np.sum((lu - lu.mean()) ** 2)
    r2 = 1.0 - np.sum((lu - pred) ** 2) / max(ss, 1e-12)
    return r2, coef[1:]


# ============================================================
# 组装：core 表达式（全用 exp）→ Nstruct → Cn
# ============================================================
def _assemble(nex, Cn, core, uexpr):
    if core == "exp-1":
        cexpr = ('sub', ('exp', uexpr), ('const', 1.0))
    elif core == "1/(exp-1)":
        cexpr = ('div', ('const', 1.0),
                 ('sub', ('exp', uexpr), ('const', 1.0)))
    elif core == "1/(2cosh)":
        neg = ('mul', ('const', -1.0), uexpr)
        cexpr = ('div', ('const', 1.0),
                 ('add', ('exp', uexpr), ('exp', neg)))
    else:  # tanh
        neg = ('mul', ('const', -1.0), uexpr)
        cexpr = ('div',
                 ('sub', ('exp', uexpr), ('exp', neg)),
                 ('add', ('exp', uexpr), ('exp', neg)))
    expr = cexpr
    if nex != ("const", 1.0):
        expr = ('mul', nex, expr)
    if abs(Cn - 1.0) > 1e-9:
        expr = ('mul', ('const', float(Cn)), expr)
    return expr


# ============================================================
# 幂次 Nstruct（阶段2）：非零列<=3、|幂|<=3、|幂|和<=6
# ============================================================
POW_MAXCOLS = 3
POW_MAXPOW = 3
POW_MAXSUM = 6
POW_FUNNEL = 0.02


def _one_factor(ce, j, p):
    if p == 1:
        return ce[j]
    if p > 1:
        return ("pow", ce[j], p)
    if p == -1:
        return ("div", ("const", 1.0), ce[j])
    return ("div", ("const", 1.0), ("pow", ce[j], -p))


def gen_power_structs(X, ce):
    """幂次单项式 Nstruct（含单列幂 x^2/x^3、三列幂次组合）。"""
    d = X.shape[1]
    out = []
    for nc in range(1, POW_MAXCOLS + 1):
        for colset in combinations(range(d), nc):
            def rec(i, cur, s):
                if i == nc:
                    nval = np.ones(len(X))
                    tup = None
                    for j, p in zip(colset, cur):
                        nval = nval * np.power(X[:, j], p)
                        f = _one_factor(ce, j, p)
                        tup = f if tup is None else ("mul", tup, f)
                    out.append((nval, tup))
                    return
                for p in range(-POW_MAXPOW, POW_MAXPOW + 1):
                    if p == 0 or s + abs(p) > POW_MAXSUM:
                        continue
                    cur.append(p)
                    rec(i + 1, cur, s + abs(p))
                    cur.pop()
            rec(0, [], 0)
    return out


def _solve_one(X, y, ce, nval, nex, core):
    """对给定 Nstruct、core 精确优化 Cn，返回 (r2, expr) 或 None。"""
    z = y / np.clip(nval, 1e-9, None)
    rel = _reliable(core, z)
    if rel.sum() < _MIN_REL:
        return None
    Xr, yr, nr = X[rel], y[rel], nval[rel]

    def obj(Cn):
        if Cn <= 0:
            return 1e9
        if np.any(np.abs(Xr) < 1e-9):
            return 1e9
        u = _invert(core, yr, nr, Cn)
        if u is None or not np.all(np.isfinite(u)) \
                or np.any(np.abs(u) < 1e-9):
            return 1e9
        r2, k = _logreg_cont(Xr, u)
        dev = np.sum((k - np.round(k)) ** 2)
        return (1.0 - r2) + 0.5 * dev

    # Cn：对数粗网格定位（连续 golden 在宽 bounds 上初始点太远收不到），
    # 再在最优粗点附近窄区间连续细化。
    grid = np.concatenate([np.logspace(-2, 2, 21), [1.0]])
    og = []
    for c in grid:
        try:
            v = obj(float(c))
        except Exception:
            v = 1e9
        if v < 1e8:
            og.append((v, float(c)))
    Cn_list = [1.0]
    if og:
        _, c0 = min(og, key=lambda t: t[0])
        try:
            res = minimize_scalar(obj, bounds=(c0 * 0.5, c0 * 2.0),
                                  method="bounded",
                                  options={"xatol": 1e-10})
            Cn_list.append(res.x)
        except Exception:
            pass
        Cn_list.append(c0)
    for Cn in Cn_list:
        u = _invert(core, y, nval, Cn)
        if u is None:
            continue
        rr = MLS.solve_mult_log(Xr, u[rel], ce)
        if rr is None or rr[0] < SOLVE_R2:
            continue
        uall = _ev_c(rr[1], X)
        if not np.all(np.isfinite(uall)):
            continue
        pred = Cn * nval * _core_val(core, uall)
        r2 = float(SC.r2(y, pred))
        if r2 >= SOLVE_R2:
            return r2, _assemble(nex, Cn, core, rr[1])
    return None


def find_power_algebra(X, y, ce):
    """Stage 2: power Nstruct, cheap screening (coarse Cn inversion + fit
    log u), then exact Cn optimization for those passing; called only after
    the stage-1 simple Nstruct fails."""
    cngrid = np.logspace(-3, 0, 7)
    cand = []
    for nval, nex in gen_power_structs(X, ce):
        z = y / np.clip(nval, 1e-9, None)
        if np.any(z <= 0):
            continue
        for core in _CORES:
            best = None
            for Cn in cngrid:
                u = _invert(core, y, nval, float(Cn))
                if u is None or np.any(~np.isfinite(u)) \
                        or np.all(np.abs(u) < 1e-9):
                    continue
                r2, k = _logreg_cont(X, u)
                dev = np.sum((k - np.round(k)) ** 2)
                s = (1.0 - r2) + 0.5 * dev
                if best is None or s < best:
                    best = s
            if best is not None and best < POW_FUNNEL:
                cand.append((best, nval, nex, core))
    cand.sort(key=lambda t: t[0])
    for _, nval, nex, core in cand:
        r = _solve_one(X, y, ce, nval, nex, core)
        if r is not None:
            return r
    return None


def find_algebra(X, y, ce):
    """返回 (r2, expr) 或 None。阶段1 简单 Nstruct；阶段2 幂次 Nstruct。"""
    if X.shape[1] > ALG_MAX_D:
        return None
    for nval, nex in _norm_structs(X, ce):
        for core in _CORES:
            r = _solve_one(X, y, ce, nval, nex, core)
            if r is not None:
                return r
    return find_power_algebra(X, y, ce)
