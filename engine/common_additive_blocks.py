# -*- coding: utf-8 -*-
"""A 类攻坚：共有乘性因子 + 加性（G1）/ exp 外层（G2）。

通用结构（不读题名/真值，探针仅排序，严格后验 r2=1）：
  G1  y = C(x)·(a1·t1 + a2·t2)，C 为共有乘性因子，t1/t2 为单项式；
  G2  y = x_j·exp(sgn·∏ x_i^p_i)，遍历变量试做乘性因子。

能力已在独立脚本（J10/_add_v3.py、_g2_solve.py）验证，本模块转为标准
block 接口 find_xxx(X,y,col_expr) -> (r2, expr)，供 fit 统一调用。
"""
import numpy as np

import shr_collapse as SC
from auto_decompose_v5 import _ev_c

COMMON_TOL = 0.08
COEF_TOL = 0.05
SOLVE_R2 = 1 - 1e-7
POWER_MAX = 6
SPOWS = (1, 2, 3, -1, -2)
_EPS = 1e-9


def _log_coef(X, y):
    ok = np.abs(y) > _EPS
    for j in range(X.shape[1]):
        ok &= np.abs(X[:, j]) > _EPS
    A = np.column_stack([np.ones(ok.sum()), np.log(np.abs(X[ok]))])
    return np.linalg.lstsq(A, np.log(np.abs(y[ok])), rcond=None)[0][1:]


def _dm(v):
    c = v - v.mean()
    s = c.std()
    return c / s if s > 1e-12 else c * 0


def _one_factor(ce, j, p):
    if p == 1:
        return ce[j]
    if p > 1:
        return ('pow', ce[j], p)
    if p == -1:
        return ('div', ('const', 1.0), ce[j])
    return ('div', ('const', 1.0), ('pow', ce[j], -p))


def _mono_expr(ce, jp):
    """jp: list of (原始列j, 幂p) 非零，左结合 mul。"""
    ex = _one_factor(ce, jp[0][0], jp[0][1])
    for j, p in jp[1:]:
        ex = ('mul', ex, _one_factor(ce, j, p))
    return ex


def _with_coef(t, c):
    if abs(c - 1.0) < 1e-7:
        return t
    if abs(c + 1.0) < 1e-7:
        return ('mul', ('const', -1.0), t)
    return ('mul', ('const', float(c)), t)


def _mult_structure(X, y):
    k = _log_coef(X, y)
    ki = [int(round(v)) for v in k]
    if any(abs(v - r) > COEF_TOL or abs(r) > POWER_MAX
           for v, r in zip(k, ki)):
        return None
    return ki


def _build_mono_vals(X, powers):
    S = np.ones(X.shape[0])
    for j, p in enumerate(powers):
        if p != 0:
            S = S * np.abs(X[:, j]) ** p
    return S


def find_common_additive(X, y, col_expr):
    """G1：y = C·(a1·t1 + a2·t2)。返回 (r2, expr) 或 None。"""
    d = X.shape[1]
    k = _log_coef(X, y)
    common = []
    for i, v in enumerate(k):
        p = int(round(v))
        if p != 0 and abs(v - p) < COMMON_TOL:
            common.append((i, p))
    if not common:
        return None
    Cvar = np.ones(X.shape[0])
    for i, p in common:
        Cvar = Cvar * np.abs(X[:, i]) ** p
    w = y / Cvar
    cands = []
    for i in range(d):
        if any(i == ci for ci, _ in common):
            continue
        for p in SPOWS:
            v = np.abs(X[:, i]) ** p
            cands.append((i, p, v, abs(np.dot(_dm(v), _dm(w)) / len(w))))
    cands.sort(key=lambda t: -t[3])
    for i, p, sv, corr in cands:
        denom = np.dot(sv - sv.mean(), sv - sv.mean())
        if denom < 1e-12:
            continue
        slope = np.dot(sv - sv.mean(), w - w.mean()) / denom
        resid = w - slope * sv
        # 初始 slope 是 OLS，受首项/残差项样本协方差污染，样本量小会偏离
        # 真实系数，残差幂次因此不严格整数（但仍接近真实结构）。不直接放弃：
        # 宽容取整得 t2 候选，联合最小二乘重新估计系数，严格 r2=1 门判定。
        for powers in _t2_candidates(_log_coef(X, resid)):
            t2v = _build_mono_vals(X, powers)
            B = np.column_stack([sv, t2v])
            cc, _, _, _ = np.linalg.lstsq(B, w, rcond=None)
            if float(SC.r2(w, B @ cc)) < SOLVE_R2:
                continue
            c_expr = _mono_expr(col_expr, common)
            term1 = _with_coef(_one_factor(col_expr, i, p), cc[0])
            rp = [(j, q) for j, q in enumerate(powers) if q]
            if not rp:
                continue
            term2 = _with_coef(_mono_expr(col_expr, rp), cc[1])
            A = ('add', term1, term2)
            full = ('mul', c_expr, A)
            pred = _ev_c(full, X)
            if not np.all(np.isfinite(pred)):
                continue
            r = float(SC.r2(y, pred))
            if r >= SOLVE_R2:
                return r, full
    return None


def _t2_candidates(rk):
    """残差 log 幂次 -> t2 候选幂次（list）。宽容取整 + 不确定位 ±1。
    错误结构由联合最小二乘的严格 r2=1 门拦住，这里只负责供候选。"""
    base = [int(round(v)) for v in rk]
    if any(abs(q) > POWER_MAX for q in base):
        return []
    out = [base]
    # 偏离取整较大的位（污染重），补 floor/ceil 邻近
    for j, v in enumerate(rk):
        if abs(v - round(v)) > 0.3 and abs(v) < POWER_MAX:
            for q in (int(np.floor(v)), int(np.ceil(v))):
                if q != base[j] and abs(q) <= POWER_MAX:
                    alt = list(base)
                    alt[j] = q
                    if alt not in out:
                        out.append(alt)
    return out


def find_common_exp(X, y, col_expr):
    """G2：y = x_j·exp(sgn·mono)。返回 (r2, expr) 或 None。"""
    if not (y > 0).all():
        return None
    d = X.shape[1]
    for j in range(d):
        keep = [i for i in range(d) if i != j]
        z = y / X[:, j]
        u = np.log(z)
        if not (u > 0).all() and not (u < 0).all():
            continue
        res = _mult_structure(X[:, keep], np.abs(u))
        if res is None:
            continue
        mono = _build_mono_vals(X[:, keep], res)
        sgn = -1.0 if (u < 0).all() else 1.0
        if float(SC.r2(u, sgn * mono)) < SOLVE_R2:
            continue
        rp = [(keep[t], q) for t, q in enumerate(res) if q]
        inner = _with_coef(_mono_expr(col_expr, rp), sgn)
        full = ('mul', col_expr[j], ('exp', inner))
        pred = _ev_c(full, X)
        if not np.all(np.isfinite(pred)):
            continue
        r = float(SC.r2(y, pred))
        if r >= SOLVE_R2:
            return r, full
    return None
