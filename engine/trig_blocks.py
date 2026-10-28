# -*- coding: utf-8 -*-
"""J5 C 三角几何复合（第一部分：反三角外层）。

结构：y = inv_trig(u)，inv_trig ∈ {arcsin, arccos, arctan}，
u 为扩展列（原始列 + sin(xi) + cos(xi)）上的单项式（含比值）。

机理：
  1) 反演 z = sin(y) / cos(y) / tan(y)；
  2) 在扩展列（原始 + sin(xi) + cos(xi)）上跑 MLS（mult_log_solver），
     幂次须干净整数、全量 r2 严格门；
  3) 值域门保证 y 在主值范围（arcsin y∈[-π/2,π/2]、arccos y∈[0,π]），
     组装后 y = inv_trig(z) 唯一。

探针是评估，真值靠全量后验 + 独立 truth 核验；无逐题分支。
"""
import numpy as np
from fractions import Fraction
import mult_log_solver as MLS

TRIG_MAX_D = 6
_EPS = 1e-9
SOLVE_R2 = 1.0 - 1e-7


def _clean_c(v, maxq=12):
    return Fraction(float(v)).limit_denominator(maxq)


def _cos_terms(X, ce):
    """枚举余弦角度项 (data, ce)：单列 cos(xa)；角度差 cos(xa-xb)。"""
    d = X.shape[1]
    out = []
    for a in range(d):
        out.append((np.cos(X[:, a]), ('cos', ce[a])))
    for a in range(d):
        for b in range(a + 1, d):
            out.append((np.cos(X[:, a] - X[:, b]),
                        ('cos', ('sub', ce[a], ce[b]))))
    return out


def find_cosine_law(X, y, ce):
    """余弦定理 y = sqrt(a²+b²-2ab cos)。返回 (r2, expr) 或 None。"""
    d = X.shape[1]
    if not (np.all(y >= 0) or np.all(y <= 0)):
        return None
    z = y ** 2
    cterms = _cos_terms(X, ce)
    edges = [(i, j) for i in range(d) for j in range(i + 1, d)]
    best = None
    for cd, cce in cterms:
        for i, j in edges:
            M = np.column_stack([X[:, i] ** 2, X[:, j] ** 2,
                                 X[:, i] * X[:, j] * cd])
            coef, _, _, _ = np.linalg.lstsq(M, z, rcond=None)
            pred = M @ coef
            r2 = 1 - np.sum((z - pred) ** 2) / max(
                np.sum((z - z.mean()) ** 2), 1e-12)
            if r2 < SOLVE_R2:
                continue
            c1, c2, c3 = coef
            if abs(c1 - c2) > max(0.05, 0.05 * abs(c1)):
                continue
            r3 = c3 / c1 if abs(c1) > 1e-9 else 0.0
            f3 = _clean_c(r3)
            if abs(float(f3) - r3) > 0.02 or abs(float(f3) + 2) > 0.02:
                continue
            f1 = _clean_c(c1)
            if abs(float(f1) - c1) > 0.02 * max(1, abs(c1)):
                continue
            xi, xj = ce[i], ce[j]
            s = ('add',
                 ('add', ('pow', xi, 2), ('pow', xj, 2)),
                 ('mul', ('const', -2),
                  ('mul', ('mul', xi, xj), cce)))
            expr = ('sqrt', s)
            if float(f1) != 1:
                expr = ('mul', ('const', float(f1)), expr)
            rp = _post_r2(X, y, expr)
            if rp is not None and rp >= SOLVE_R2:
                if best is None or rp > best[0]:
                    best = (rp, expr)
    return best


def _quick_log_r2(X, r):
    """Bare log-regression score (no rounding, non-strict): returns the fit
    r2 for staged screening."""
    ok = np.abs(r) > 1e-9
    for j in range(X.shape[1]):
        ok &= np.abs(X[:, j]) > 1e-9
    if ok.sum() < X.shape[1] + 2:
        return -1.0
    A = np.column_stack([np.ones(ok.sum()), np.log(np.abs(X[ok]))])
    coef = np.linalg.lstsq(A, np.log(np.abs(r[ok])), rcond=None)[0]
    pred = A @ coef
    ss = np.sum((np.log(np.abs(r[ok])) - np.log(np.abs(r[ok])).mean()) ** 2)
    return float(1 - np.sum((np.log(np.abs(r[ok])) - pred) ** 2) / max(ss, 1e-12))


def find_cosine_law_reciprocal(X, y, ce):
    """余弦定理在分母/分子，外层乘性：y = A / sqrt(余弦) 或 A·sqrt(余弦)。

    Screening: build the cosine structure (edge pairs + angle, directly
    1,1,-2); after peeling, first score with bare log regression (fast),
    and only run solve_mult_log strict verification for high-score candidates.
    返回 (r2, expr) 或 None。
    """
    d = X.shape[1]
    cterms = _cos_terms(X, ce)
    edges = [(i, j) for i in range(d) for j in range(i + 1, d)]
    cands = []
    for cd, cce in cterms:
        for i, j in edges:
            sval = (X[:, i] ** 2 + X[:, j] ** 2
                    - 2 * X[:, i] * X[:, j] * cd)
            if np.any(sval <= 1e-9):
                continue
            cl = np.sqrt(sval)
            xi, xj = ce[i], ce[j]
            s = ('add', ('add', ('pow', xi, 2), ('pow', xj, 2)),
                 ('mul', ('const', -2), ('mul', ('mul', xi, xj), cce)))
            cl_expr = ('sqrt', s)
            for kind, r, shape in (
                    ('div', y * cl,
                     ('mul', None, ('div', ('const', 1), cl_expr))),
                    ('mul', y / np.clip(cl, 1e-9, None),
                     ('mul', None, cl_expr))):
                q = _quick_log_r2(X, r)
                if q > 0.98:
                    cands.append((q, r, shape))
    cands.sort(key=lambda t: -t[0])
    best = None
    for q, r, shape in cands[:8]:
        ok = np.isfinite(r) & (np.abs(r) > 1e-9)
        if ok.sum() < d + 3:
            continue
        res = MLS.solve_mult_log(X[ok], r[ok], ce)
        if res is None or res[0] < SOLVE_R2:
            continue
        expr = (shape[0], res[1], shape[2])
        rp = _post_r2(X, y, expr)
        if rp is not None and rp >= SOLVE_R2:
            if best is None or rp > best[0]:
                best = (rp, expr)
    return best


def _trig_extended(X, ce):
    """扩展列：原始列 + sin(xi) + cos(xi)。返回 (Xext, ceext)。"""
    d = X.shape[1]
    cols = [X[:, j] for j in range(d)]
    ces = [ce[j] for j in range(d)]
    for j in range(d):
        cols.append(np.sin(X[:, j]))
        ces.append(('sin', ce[j]))
    for j in range(d):
        cols.append(np.cos(X[:, j]))
        ces.append(('cos', ce[j]))
    return np.column_stack(cols), ces


def find_inverse_trig(X, y, ce):
    """反三角外层 y = inv_trig(u)。返回 (r2, expr_tuple) 或 None。"""
    Xext, ceext = _trig_extended(X, ce)
    candidates = [
        ('asin', np.sin(y),
         np.all(np.abs(y) <= np.pi / 2 + 0.05)),
        ('acos', np.cos(y),
         np.all(y >= -0.05) and np.all(y <= np.pi + 0.05)),
        ('atan', np.tan(y), True),
    ]
    best = None
    for inv, z, range_ok in candidates:
        if not range_ok:
            continue
        res = MLS.solve_mult_log(Xext, z, ceext)
        if res is None:
            continue
        r, inner, C = res
        if r < SOLVE_R2:
            continue
        expr = (inv, inner)
        # 组装后全量数值后验（防 arcsin 多值/出域）
        rp = _post_r2(X, y, expr)
        if rp is not None and rp >= SOLVE_R2:
            if best is None or rp > best[0]:
                best = (rp, expr)
    return best


def _post_r2(X, y, expr):
    """组装后 y 的全量 r2（用引擎求值器）。"""
    try:
        import auto_decompose_v5 as AD
        yp = AD._ev_c(expr, X)
        yp = np.asarray(yp, dtype=float)
        ss = np.sum((y - y.mean()) ** 2)
        return float(1 - np.sum((y - yp) ** 2) / max(ss, 1e-12))
    except Exception:
        return None


def _mobius_components(X, ce):
    """Möbius 分式构件库：cos 列、双向比（有序）。返回 (data, expr) 列表。"""
    d = X.shape[1]
    out = []
    for k in range(d):
        out.append((np.cos(X[:, k]), ('cos', ce[k])))
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            if np.all(np.abs(X[:, j]) > 1e-8):
                out.append((X[:, i] / X[:, j], ('div', ce[i], ce[j])))
    return out


def find_inverse_mobius(X, y, ce):
    """反三角（主 acos）包裹的 Möbius 分式：y = acos((a±b)/(1±ab))。

    枚举两个构件（cos 列 / 双向比），试 (a−b)/(1−ab)、(a+b)/(1+ab)，
    对 cos(y) 拟合；命中后全量后验。返回 (r2, expr) 或 None。
    """
    comps = _mobius_components(X, ce)
    targets = [('acos', np.cos(y)), ('asin', np.sin(y))]
    best = None
    for outer, z in targets:
        for ai, (a, ae) in enumerate(comps):
            for bi, (b, be) in enumerate(comps):
                if ai == bi:
                    continue
                ab = a * b
                forms = []
                if np.all(np.abs(1 - ab) > 1e-9):
                    forms.append(('div', ('sub', ae, be),
                                  ('sub', ('const', 1), ('mul', ae, be)),
                                  (a - b) / (1 - ab)))
                if np.all(np.abs(1 + ab) > 1e-9):
                    forms.append(('div', ('add', ae, be),
                                  ('add', ('const', 1), ('mul', ae, be)),
                                  (a + b) / (1 + ab)))
                for _, num, den, fv in forms:
                    if not np.all(np.isfinite(fv)):
                        continue
                    q = 1 - np.sum((z - fv) ** 2) / max(
                        np.sum((z - z.mean()) ** 2), 1e-12)
                    if q < SOLVE_R2:
                        continue
                    expr = (outer, ('div', num, den))
                    rp = _post_r2(X, y, expr)
                    if rp is not None and rp >= SOLVE_R2:
                        if best is None or rp > best[0]:
                            best = (rp, expr)
    return best


WAVE_MAX_DEPTH = 2
_STRIP_EPS = 1e-6


def _h2(a, b):
    """半角表达式 a*b/2（tuple）。"""
    return ('mul', ('const', 0.5), ('mul', a, b))


def _h1(a):
    """半角表达式 a/2（tuple）。"""
    return ('mul', ('const', 0.5), a)


def _slit_candidates(X, ce):
    """多缝因子 f=sin²(nθ/2)/sin²(θ/2)。返回 (f_data, f_expr) 列表。"""
    d = X.shape[1]
    out = []
    for k in range(d):
        hk = 0.5 * X[:, k]
        den = np.sin(hk)
        if np.any(np.abs(den) < 1e-8):
            continue
        for m in range(d):
            if m == k:
                continue
            hmk = 0.5 * X[:, m] * X[:, k]
            f = np.sin(hmk) ** 2 / den ** 2
            f_expr = ('div', ('pow', ('sin', _h2(ce[m], ce[k])), 2),
                      ('pow', ('sin', _h1(ce[k])), 2))
            out.append((f, f_expr))
    return out


def _sinc2_candidates(X, ce):
    """sinc² 因子 sin²(φ)/φ²。返回 (s2_data, s2_expr) 列表。"""
    d = X.shape[1]
    phis = []
    for i in range(d):
        phis.append((0.5 * X[:, i], _h1(ce[i])))
    for i in range(d):
        for j in range(i + 1, d):
            phis.append((0.5 * X[:, i] * X[:, j], _h2(ce[i], ce[j])))
            for k in range(d):
                if k == i or k == j:
                    continue
                phi = 0.5 * (X[:, i] - X[:, j]) * X[:, k]
                phi_e = ('mul', ('const', 0.5),
                         ('mul', ('sub', ce[i], ce[j]), ce[k]))
                phis.append((phi, phi_e))
    out = []
    for phi, phi_e in phis:
        s2 = np.where(np.abs(phi) < 1e-9, 1.0,
                      np.sin(phi) ** 2 / phi ** 2)
        s2_expr = ('div', ('pow', ('sin', phi_e), 2),
                   ('pow', phi_e, 2))
        out.append((s2, s2_expr))
    return out


def find_wave_interference(X, y, ce, depth=0):
    """波干涉递归反演：y = A·[多缝因子]·[sinc² 因子]…，A 为干净乘性。

    每层先试纯 MLS（底）；失败则剥一个因子（多缝 / sinc²）后递归。
    depth 上限 WAVE_MAX_DEPTH；返回 (r2, expr) 或 None。
    """
    base = MLS.solve_mult_log(X, y, ce)
    if base is not None and base[0] >= SOLVE_R2:
        return (base[0], base[1])
    if depth >= WAVE_MAX_DEPTH:
        return None
    factors = _slit_candidates(X, ce) + _sinc2_candidates(X, ce)
    best = None
    for fdata, fexpr in factors:
        ok = np.abs(fdata) > _STRIP_EPS
        if ok.sum() < X.shape[1] + 3:
            continue
        r = y[ok] / fdata[ok]
        inner = find_wave_interference(X[ok], r, ce, depth + 1)
        if inner is None:
            continue
        expr = ('mul', inner[1], fexpr)
        rp = _post_r2(X, y, expr)
        if rp is not None and rp >= SOLVE_R2:
            if best is None or rp > best[0]:
                best = (rp, expr)
    return best


def find_interference_additive(X, y, ce):
    """加性干涉 y = Ia + Ib + 2 sqrt(Ia Ib) cos delta（返回 (r2, expr) 或 None）。

    通用枚举：角度列 k、两个正强度列 i,j；构造 cross=sqrt(Ia Ib) cos delta；
    加性列 [Ia, Ib, cross] 上 lstsq，系数须为 (1,1,2)，全量 r2 严格门。
    """
    d = X.shape[1]
    best = None
    for k in range(d):
        cd = np.cos(X[:, k])
        for i in range(d):
            if i == k or not np.all(X[:, i] > 0):
                continue
            for j in range(i + 1, d):
                if j == k or not np.all(X[:, j] > 0):
                    continue
                cross = np.sqrt(X[:, i] * X[:, j]) * cd
                M = np.column_stack([X[:, i], X[:, j], cross])
                try:
                    coef, _, _, _ = np.linalg.lstsq(M, y, rcond=None)
                except Exception:
                    continue
                pred = M @ coef
                rv = 1 - np.sum((y - pred) ** 2) / max(
                    np.sum((y - y.mean()) ** 2), 1e-12)
                if rv < SOLVE_R2:
                    continue
                c1, c2, c3 = coef
                f1, f2, f3 = _clean_c(c1), _clean_c(c2), _clean_c(c3)
                if not (abs(float(f1) - 1) < 0.03 and
                        abs(float(f2) - 1) < 0.03 and
                        abs(float(f3) - 2) < 0.03):
                    continue
                xi, xj, xk = ce[i], ce[j], ce[k]
                cross_e = ('mul', ('sqrt', ('mul', xi, xj)), ('cos', xk))
                expr = ('add', ('add', xi, xj),
                        ('mul', ('const', 2), cross_e))
                rp = _post_r2(X, y, expr)
                if rp is not None and rp >= SOLVE_R2:
                    if best is None or rp > best[0]:
                        best = (rp, expr)
    return best
