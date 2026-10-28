# -*- coding: utf-8 -*-
"""J3 D2 根号复合反演：通用、规则统一、不读题名/真值。

机理
----
物理公式 y = sgn * N_half / sqrt(g)：
  * 外层 sqrt（y 同号），z = y^2 = N / g；
  * 洛伦兹类分母 g = 1 - (v/c)^2（归一化，含双向比的平方），
    现有 build_denom_factors 只有 u=v/c（一次），缺 u^2=v^2/c^2；
  * 分子 N 为完全平方单项式（m0^2、(m0*v)^2、(m*c^2)^2 等），
    由纯乘性 log 回归（mult_log_solver）一次性求幂次，取 sqrt 后
    把分子提到根号外（结构与 truth 同构，避免 sympy 不提取 sqrt(x^2)）。

Staged screening (no enumeration explosion):
  1. 门：y 同号（全正/全负），取 sgn；
  2. 对每一对列 (a,b) 统一构造 g = 1 - (x_a/x_b)^2；
  3. w = z*g = y^2*g，用 MLS 验证为单项式并取常数 C，再求整数幂次 ki；
  4. 要求 w 完全平方（ki 全偶），取半幂 ki/2 与 sqrt(C)，构造 N_half；
  5. 组装 y = sgn*N_half/sqrt(g)，在原始 y 后验严格门。

加性分子（如 (1+v/c)、t-u*x/c^2、x-u*t）不在 J3：
  MLS 只解单项式，加性分子留 J6（加性递归分解）。

符号 exact 由独立符号等价核验兜底（高 r2 但结构错不判 exact）。
"""
import numpy as np

import mult_log_solver as MLS
from mult_log_solver import COEF_TOL, POWER_MAX
from shr_v7 import SOLVE_R2
import shr_collapse as SC
from auto_decompose_v5 import _ev_c

_EPS = 1e-9


def _powers_w(X, w):
    """log 回归求 w 的整数幂次，返回 ki 或 None。"""
    ok = np.abs(w) > _EPS
    for j in range(X.shape[1]):
        ok &= np.abs(X[:, j]) > _EPS
    Xo, wo = X[ok], w[ok]
    if len(wo) < X.shape[1] + 2:
        return None
    A = np.column_stack([np.ones(len(wo)), np.log(np.abs(Xo))])
    coef = np.linalg.lstsq(A, np.log(np.abs(wo)), rcond=None)[0]
    k = coef[1:]
    ki = [int(round(v)) for v in k]
    if any(abs(v - r) > COEF_TOL or abs(r) > POWER_MAX * 2
           for v, r in zip(k, ki)):
        return None
    return ki


def _half_monomial(X, col_expr, half, c0):
    """半幂单项式：返回 (vec, expr)。"""
    d = X.shape[1]
    vec = np.ones(X.shape[0])
    terms = []
    for j, hj in enumerate(half):
        if hj == 0:
            continue
        ax = np.abs(X[:, j])
        if hj % 2 == 0:
            vec = vec * np.power(ax, hj)
        else:
            vec = vec * np.sign(X[:, j]) * np.power(ax, hj)
        if hj == 1:
            terms.append(col_expr[j])
        elif hj > 0:
            terms.append(('pow', col_expr[j], hj))
        elif hj == -1:
            terms.append(('div', ('const', 1.0), col_expr[j]))
        else:
            terms.append(('div', ('const', 1.0), ('pow', col_expr[j], -hj)))
    vec = vec * c0
    expr = ('const', float(c0))
    for t in terms:
        expr = ('mul', expr, t)
    return vec, expr


def find_outer_sqrt(X, y, col_expr):
    """返回 (r2, expr_tuple) 或 None。对所有列对同一规则，不逐题打补丁。"""
    if not (np.all(y > 0) or np.all(y < 0)):
        return None
    d = X.shape[1]
    sgn = 1 if np.mean(y) > 0 else -1
    z = y ** 2
    best = None

    for a in range(d):
        for b in range(d):
            if a == b:
                continue
            xb = np.clip(X[:, b], _EPS, None)
            ratio = X[:, a] / xb
            g = 1.0 - ratio ** 2
            if not np.all(np.isfinite(g)) or np.std(g) < 1e-12:
                continue
            g_expr = ('add', ('const', 1.0),
                      ('mul', ('const', -1.0),
                       ('pow', ('div', col_expr[a], col_expr[b]), 2)))
            w = z * g
            if not np.all(np.isfinite(w)) or np.std(w) < 1e-12:
                continue
            sol = MLS.solve_mult_log(X, w, col_expr)
            if sol is None:
                continue
            _, _, C = sol
            ki = _powers_w(X, w)
            if ki is None:
                continue
            # w 必须完全平方（变量幂次全偶、常数 C 非负）
            if any(kj % 2 != 0 for kj in ki) or C <= 0:
                continue
            half = [kj // 2 for kj in ki]
            c0 = np.sqrt(C)
            nvec, nexpr = _half_monomial(X, col_expr, half, c0)
            gv = _ev_c(g_expr, X)
            denom = np.sqrt(np.clip(gv, _EPS, None))
            ypred = sgn * nvec / denom
            if np.any(gv < -1e-6):
                continue
            r = float(SC.r2(y, ypred))
            if r >= SOLVE_R2:
                expr = ('div', nexpr, ('sqrt', g_expr))
                if sgn < 0:
                    expr = ('mul', ('const', -1.0), expr)
                if best is None or r > best[0]:
                    best = (r, expr)
    return best
