# -*- coding: utf-8 -*-
"""J6 B 加性递归分解：通用、规则统一、不读题名/真值。

机理
----
物理公式 y = sgn * A(x) / sqrt(g)，分子 A 为**加性表达式**（不是单项式）：
  * 洛伦兹变换：x1 = (x-u*t)/sqrt(1-u^2/c^2)，
    t1 = (t-u*x/c^2)/sqrt(1-u^2/c^2)，
    omega = (1+v/c)*omega_0/sqrt(1-v^2/c^2)。
  * 分母 g = 1-(x_a/x_b)^2（同 J3 的平方比值，框架复用）。
  * J3 的 MLS 只解单项式分子；J6 把分子改成在 q=sqrt(w) 上做**加性列空间
    线性求解**（常数/原始/pair 积/三元积商/两积除变量平方/双向比），不展开
    平方、不做盲目搜索。

Staged screening
  1. Gate: y same sign (sgn), z=y^2;
  2. 对每列对 (a,b) 统一造 g=1-(x_a/x_b)^2，w=z*g，q=sqrt(w)；
  3. 加性列空间（QR 主元去共线）最小二乘 → 系数保守有理化（Fraction q<=12）；
  4. 用清理后系数重建 q，严格 r2 门才接受，组装加性 A；
  5. y=sgn*A/sqrt(g)，原始 y 后验严格门。

d>=6 完整加性列空间爆炸，设 ADDITIVE_MAX_D=5，d>=6 不跑完整加性。
符号 exact 由独立符号等价核验兜底。
"""
import numpy as np
from itertools import combinations, permutations
from fractions import Fraction

from shr_v7 import SOLVE_R2
import shr_collapse as SC
from auto_decompose_v5 import _ev_c

_EPS = 1e-9
ADDITIVE_MAX_D = 5
_QMAX = 12          # 有理化分母上限
_ZERO_TOL = 0.03    # 系数判 0 上限


def _add_space(X, ce):
    """加性列空间：返回 (B, tups)，QR 主元去共线。"""
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    tups = [('const', 1.0)]
    for i in range(d):
        cols.append(X[:, i])
        tups.append(ce[i])
    for i, j in combinations(range(d), 2):
        cols.append(X[:, i] * X[:, j])
        tups.append(('mul', ce[i], ce[j]))
    for i, j, k in permutations(range(d), 3):
        cols.append(X[:, i] * X[:, j] / np.clip(X[:, k], _EPS, None))
        tups.append(('div', ('mul', ce[i], ce[j]), ce[k]))
    for k in range(d):
        for i, j in combinations([x for x in range(d) if x != k], 2):
            cols.append(X[:, i] * X[:, j] / np.clip(X[:, k], _EPS, None) ** 2)
            tups.append(('div', ('mul', ce[i], ce[j]), ('pow', ce[k], 2)))
    for i, j in permutations(range(d), 2):
        cols.append(X[:, i] / np.clip(X[:, j], _EPS, None))
        tups.append(('div', ce[i], ce[j]))
    B = np.column_stack(cols)
    rank = np.linalg.matrix_rank(B, tol=1e-8)
    R = np.linalg.qr(B, mode='r')
    diagR = np.abs(np.diag(R))
    order = np.argsort(-diagR)
    keep = order[:rank]
    return B[:, keep], [tups[i] for i in keep]


def _clean_coef(a):
    out = []
    for v in a:
        if abs(v) < _ZERO_TOL:
            out.append(Fraction(0))
        else:
            out.append(Fraction(float(v)).limit_denominator(_QMAX))
    return out


def _coef_term(c, t):
    if c == 1:
        return t
    if c == -1:
        return ('mul', ('const', -1.0), t)
    return ('mul', ('const', float(c)), t)


def _build_add_expr(terms):
    """terms=[(Fraction,tup)]，add 二元左结合。"""
    expr = _coef_term(terms[0][0], terms[0][1])
    for c, t in terms[1:]:
        expr = ('add', expr, _coef_term(c, t))
    return expr


def find_additive_outer_sqrt(X, y, col_expr):
    """返回 (r2, expr_tuple) 或 None。对所有列对同一规则，不逐题打补丁。"""
    if X.shape[1] > ADDITIVE_MAX_D:
        return None
    if not (np.all(y > 0) or np.all(y < 0)):
        return None
    d = X.shape[1]
    sgn = 1 if np.mean(y) > 0 else -1
    z = y ** 2
    B, tups = _add_space(X, col_expr)
    best = None

    for a in range(d):
        for b in range(d):
            if a == b:
                continue
            xb = np.clip(X[:, b], _EPS, None)
            g = 1.0 - (X[:, a] / xb) ** 2
            if not np.all(np.isfinite(g)) or np.any(g < _EPS):
                continue
            g_expr = ('add', ('const', 1.0),
                      ('mul', ('const', -1.0),
                       ('pow', ('div', col_expr[a], col_expr[b]), 2)))
            w = z * g
            q = np.sqrt(np.clip(w, _EPS, None))
            coef, _, _, _ = np.linalg.lstsq(B, q, rcond=None)
            if float(SC.r2(q, B @ coef)) < 1 - 1e-6:
                continue
            fr = _clean_coef(coef)
            qc = np.zeros_like(q)
            for i, c in enumerate(fr):
                if c != 0:
                    qc = qc + float(c) * B[:, i]
            rc = float(SC.r2(q, qc))
            if rc < SOLVE_R2:
                continue
            terms = [(c, tups[i]) for i, c in enumerate(fr) if c != 0]
            if not terms:
                continue
            Aexpr = _build_add_expr(terms)
            Avec = _ev_c(Aexpr, X)
            gv = _ev_c(g_expr, X)
            ypred = sgn * Avec / np.sqrt(np.clip(gv, _EPS, None))
            r = float(SC.r2(y, ypred))
            if r >= SOLVE_R2:
                expr = ('div', Aexpr, ('sqrt', g_expr))
                if sgn < 0:
                    expr = ('mul', ('const', -1.0), expr)
                if best is None or r > best[0]:
                    best = (r, expr)
    return best


def _inner_add_space(X, ce, include_triple=False):
    """根号内直接加性积木（去重）。

    低阶（include_triple=False）：单列/平方/倒数、单列比、平方/单列、
    平方/平方、pair。
    高阶（include_triple=True）：再加平方/两列、单列/两列、平方/双平方。

    分母无序组合用 combinations 枚举，避免排列产生重复列。
    返回原始（未标准化）积木 + tups，不做 QR；稀疏选择由求解器负责。
    """
    d = X.shape[1]
    n = X.shape[0]
    x = [np.clip(X[:, i], _EPS, None) for i in range(d)]
    cols = [np.ones(n)]
    tups = [('const', 1.0)]
    for i in range(d):
        cols += [x[i], x[i] ** 2, 1.0 / x[i], 1.0 / x[i] ** 2]
        tups += [ce[i], ('pow', ce[i], 2),
                 ('div', ('const', 1.0), ce[i]),
                 ('div', ('const', 1.0), ('pow', ce[i], 2))]
    for i, j in permutations(range(d), 2):
        cols += [x[i] / x[j], x[i] ** 2 / x[j], x[i] ** 2 / x[j] ** 2]
        tups += [('div', ce[i], ce[j]),
                 ('div', ('pow', ce[i], 2), ce[j]),
                 ('div', ('pow', ce[i], 2), ('pow', ce[j], 2))]
    for i, j in combinations(range(d), 2):
        cols.append(x[i] * x[j])
        tups.append(('mul', ce[i], ce[j]))
    if include_triple:
        from itertools import permutations as _perm
        for i in range(d):
            rest = [r for r in range(d) if r != i]
            for j, k in combinations(rest, 2):
                xj, xk = x[j], x[k]
                # x_i² / (x_j·x_k)（分母对称）
                cols.append(x[i] ** 2 / (xj * xk))
                tups.append(('div', ('pow', ce[i], 2), ('mul', ce[j], ce[k])))
                # x_i / (x_j·x_k)
                cols.append(x[i] / (xj * xk))
                tups.append(('div', ce[i], ('mul', ce[j], ce[k])))
                # x_i² / (x_j²·x_k²)
                cols.append(x[i] ** 2 / (xj ** 2 * xk ** 2))
                tups.append(('div', ('pow', ce[i], 2),
                             ('mul', ('pow', ce[j], 2), ('pow', ce[k], 2))))
        # x_a·x_b² / x_c²（alpha·c²/d² 型；a,b 幂次不同，须有序）
        for a, b, c in _perm(range(d), 3):
            cols.append(x[a] * x[b] ** 2 / x[c] ** 2)
            tups.append(('div', ('mul', ce[a], ('pow', ce[b], 2)),
                         ('pow', ce[c], 2)))
        # x_i²·x_j² / x_k²（两列平方积除一列平方：omega²·v²/c² 型）
        for k in range(d):
            rest = [r for r in range(d) if r != k]
            for i, j in combinations(rest, 2):
                cols.append(x[i] ** 2 * x[j] ** 2 / x[k] ** 2)
                tups.append(('div', ('mul', ('pow', ce[i], 2),
                                     ('pow', ce[j], 2)),
                             ('pow', ce[k], 2)))
    B = np.nan_to_num(np.column_stack(cols), nan=0.0, posinf=0.0, neginf=0.0)
    return B, tups


def _omp_select(B, z, max_terms=8, r2_tol=1e-6):
    """正交匹配追踪：标准化后逐步选与残差最相关的列，天然稀疏、抗共线。
    返回选中的列索引（原始 B）。"""
    norms = np.linalg.norm(B, axis=0)
    norms = np.where(norms < _EPS, 1.0, norms)
    Bn = B / norms
    selected = []
    avail = list(range(B.shape[1]))
    for _ in range(max_terms):
        if not avail:
            break
        resid = z - (B[:, selected] @ np.linalg.lstsq(B[:, selected], z,
                     rcond=None)[0]) if selected else z.copy()
        corr = np.abs(Bn[:, avail].T @ resid)
        j = avail[int(np.argmax(corr))]
        selected.append(j)
        avail = [a for a in avail if a != j]
        fit = B[:, selected] @ np.linalg.lstsq(B[:, selected], z, rcond=None)[0]
        if float(SC.r2(z, fit)) >= 1 - r2_tol:
            break
    return selected


def _clean_coef_mixed(a):
    """系数先 Fraction(q<=12)，偏离大（如 π²≈9.87）保留精确数值。"""
    out = []
    for v in a:
        if abs(v) < _ZERO_TOL:
            out.append(Fraction(0))
        else:
            fr = Fraction(float(v)).limit_denominator(_QMAX)
            out.append(fr if abs(float(fr) - v) < 0.002 else float(v))
    return out


def _solve_inner(z, B, tups):
    """在给定积木池上做加性 lstsq，返回 (expr, r2) 或 None。"""
    coef, _, _, _ = np.linalg.lstsq(B, z, rcond=None)
    if float(SC.r2(z, B @ coef)) < 1 - 1e-6:
        return None
    fr = _clean_coef_mixed(coef)
    zc = np.zeros_like(z)
    for i, c in enumerate(fr):
        if c != 0:
            zc = zc + float(c) * B[:, i]
    if float(SC.r2(z, zc)) < SOLVE_R2:
        return None
    terms = [(c, tups[i]) for i, c in enumerate(fr) if c != 0]
    if not terms:
        return None
    return _build_add_expr(terms)


def find_sqrt_inner_additive(X, y, col_expr):
    """y=sqrt(加性)，直接在 y² 上加性列空间（不剥 g）。

    Staged screening: low-order blocks first (within two columns), no
    higher order if solved; three-column blocks are added only when the
    low-order r2 is insufficient (avoiding near-collinear columns polluting
    the clean low-order solution).
    """
    if X.shape[1] > ADDITIVE_MAX_D:
        return None
    if not (np.all(y > 0) or np.all(y < 0)):
        return None
    sgn = 1 if np.mean(y) > 0 else -1
    z = y ** 2
    inner = None
    for include_triple in (False, True):
        B, tups = _inner_add_space(X, col_expr, include_triple=include_triple)
        inner = _solve_inner(z, B, tups)
        if inner is not None:
            break
    if inner is None:
        return None
    expr = ('sqrt', inner)
    if sgn < 0:
        expr = ('mul', ('const', -1.0), expr)
    pred = _ev_c(expr, X)
    if not np.all(np.isfinite(pred)):
        return None
    r = float(SC.r2(y, pred))
    return (r, expr) if r >= SOLVE_R2 else None


def find_sum_column_mls(X, y, col_expr):
    """构造两列和（总质量/总量）后再 MLS 纯乘性。
    覆盖 2π d^1.5/sqrt(G(m1+m2))：和列作分母；半幂走 y² 再开方。"""
    import mult_log_solver as MLS
    d = X.shape[1]
    if d > ADDITIVE_MAX_D:
        return None
    sgn = 1.0 if np.all(y >= 0) else (-1.0 if np.all(y <= 0) else 0.0)
    for a, b in combinations(range(d), 2):
        for op, fv in (('add', X[:, a] + X[:, b]),
                       ('sub', X[:, a] - X[:, b])):
            if np.min(np.abs(fv)) < _EPS:
                continue
            Xext = np.column_stack([X, fv])
            ce_ext = list(col_expr) + [(op, col_expr[a], col_expr[b])]
            # 直接 y（整幂）
            for target, wrap in ((y, None), (y ** 2, 'sqrt')):
                if wrap == 'sqrt' and sgn == 0:
                    continue
                ok = np.abs(target) > _EPS
                res = MLS.solve_mult_log(Xext[ok], target[ok], ce_ext)
                if res is None:
                    continue
                inner = res[1]
                expr = ('sqrt', inner) if wrap == 'sqrt' else inner
                if wrap == 'sqrt' and sgn < 0:
                    expr = ('mul', ('const', -1.0), expr)
                pred = _ev_c(expr, Xext)
                if not np.all(np.isfinite(pred)):
                    continue
                r = float(SC.r2(y, pred))
                if r >= SOLVE_R2:
                    return r, expr
    return None
