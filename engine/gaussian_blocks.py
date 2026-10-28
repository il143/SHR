# -*- coding: utf-8 -*-
"""高斯族通用结构发现（J5 扩展）：
    y = exp(-u^2/2) / (sqrt(2*pi) * s)
宽度 s 为单列；位置项 u 为 单列/s 或 (两列差)/s。
先以 -1/2 二次关系识别（线性化 ln(y sqrt(2pi) s) = -u^2/2），再全量后验。
"""
import numpy as np
from fractions import Fraction

_EPS = 1e-9
_SOLVE_R2 = 1 - 1e-7
_SQRT2PI = float(np.sqrt(2 * np.pi))


def _location_terms(X, col_expr, s):
    """返回 (u_array, u_expr, dep)：单列/s 或 (列差)/s。"""
    d = X.shape[1]
    out = []
    for a in range(d):
        if a == s:
            continue
        out.append((X[:, a] / X[:, s],
                    ('div', col_expr[a], col_expr[s]), [a, s]))
    for a in range(d):
        for b in range(a + 1, d):
            if a == s or b == s:
                continue
            out.append(((X[:, a] - X[:, b]) / X[:, s],
                        ('div', ('sub', col_expr[a], col_expr[b]),
                         col_expr[s]), [a, b, s]))
    return out


def find_gaussian(X, y, col_expr):
    """返回 (r2, full_expr) 或 None。"""
    d = X.shape[1]
    for s in range(d):
        sv = X[:, s]
        if np.min(np.abs(sv)) < _EPS:
            continue
        lhs = y * _SQRT2PI * sv
        ok = lhs > 0
        if ok.sum() < 8:
            continue
        lz = np.log(lhs[ok])
        for u, ue, dep in _location_terms(X, col_expr, s):
            u2 = u[ok] ** 2
            A = np.column_stack([np.ones(ok.sum()), u2])
            coef = np.linalg.lstsq(A, lz, rcond=None)[0]
            c0, c1 = coef
            # 常数项应 ~0、二次系数应 -1/2
            if abs(c1 + 0.5) > 0.02 or abs(c0) > 0.05:
                continue
            inner = ('mul', ('const', -0.5), ('pow', ue, 2))
            full = ('div', ('exp', inner),
                    ('mul', ('const', _SQRT2PI), col_expr[s]))
            try:
                from auto_decompose_v5 import _ev_c
                pred = _ev_c(full, X)
            except Exception:
                continue
            if not np.all(np.isfinite(pred)):
                continue
            ss = np.sum((y - y.mean()) ** 2)
            r = float(1 - np.sum((y - pred) ** 2) / max(ss, 1e-12))
            if r >= _SOLVE_R2:
                return r, full
    return None
