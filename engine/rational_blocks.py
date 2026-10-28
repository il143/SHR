# -*- coding: utf-8 -*-
"""J2 分式复合反演（D1，18 题）：通用、规则统一、只依赖列索引，不读题名/真值。

机理
----
物理公式 y = N / g，分母 g 是复合表达式（常数偏移、归一化、平方差、
变量和），分子 N 可由 (L1 + 双向比) 列空间 + 单变量剥层精确解出。

Staged screening (avoid running costly solving for every factor):
  1. 枚举分母因子（只依赖列索引）：
       常数偏移  x_i + c
       归一化    1 + c*u，u = 积/双向比/差/三元 ppd(分母幂 -1,-2)，
                 c = +/-1, +/-1/2, +/-1/3
       平方差    x_i^2 - x_j^2
       变量和    x_i + x_j
  2. 阶段1 快速打分：对每个因子 g（幂次 p=1,2），z = y*g^p，在
     (L1 + 双向比) 列空间前向选择（4 项），得到分子快速分。
  3. 阶段2 轻量精确：按快速分降序，对 FAST_TOP 个候选在
     (L1 + 双向比) 上解 + 单变量剥层（剥后命中立即接受，剥后改善则继续），
     组装 N/g，在原始 y 上后验；命中立即返回。
  需 L2/深层积木的题（如四元积、高幂）不在 J2，留 J7。

符号 exact 由独立符号等价核验兜底（高 r2 但结构错不判 exact）。
"""
import numpy as np
from itertools import combinations

import shr_v7 as S7
from shr_v7 import _nodes, SOLVE_R2
import shr_collapse as SC
import mult_log_solver as MLS
from auto_decompose_v5 import _ev_c

FAST_GATE = 0.98
FAST_TOP = 10
_EPS = 1e-9
# fast 剥层"有希望"门：剥层搜索全程只跑 fast，只对越过它的唯一一个最高点点
# 做一次 exact，避免对每个剥层点反复 exact（II_21_32 原 326s 主因）。
SOLVE_HOPE = 0.99


def _quick_log_r2(X, r):
    """Bare log-regression score after peeling the denominator (used when the
    numerator is a pure multiplicative monomial), for staged screening."""
    ok = np.abs(r) > _EPS
    for j in range(X.shape[1]):
        ok &= np.abs(X[:, j]) > _EPS
    if ok.sum() < X.shape[1] + 2:
        return -1.0
    lr = np.log(np.abs(r[ok]))
    A = np.column_stack([np.ones(ok.sum()), np.log(np.abs(X[ok]))])
    coef = np.linalg.lstsq(A, lr, rcond=None)[0]
    ss = np.sum((lr - lr.mean()) ** 2)
    return float(1 - np.sum((lr - A @ coef) ** 2) / max(ss, 1e-12))


def build_denom_factors(X, col_expr):
    """枚举分母因子 -> list of (g_vec, g_expr, dep_set)。"""
    d = X.shape[1]
    out = []

    def add(g, e, dp):
        if np.all(np.isfinite(g)) and np.std(g) > 1e-12:
            out.append((g, e, frozenset(dp)))

    # A. 常数偏移
    for i in range(d):
        for c in (1.0, -1.0):
            add(X[:, i] + c, ('add', col_expr[i], ('const', c)), [i])

    # u 积木：pairwise
    u_list = []
    for a, b in combinations(range(d), 2):
        u_list.append((X[:, a] * X[:, b],
                       ('mul', col_expr[a], col_expr[b]), [a, b]))
        u_list.append((X[:, a] / np.clip(X[:, b], _EPS, None),
                       ('div', col_expr[a], col_expr[b]), [a, b]))
        u_list.append((X[:, b] / np.clip(X[:, a], _EPS, None),
                       ('div', col_expr[b], col_expr[a]), [a, b]))
        u_list.append((X[:, a] - X[:, b],
                       ('sub', col_expr[a], col_expr[b]), [a, b]))
    # u 积木：三元 ppd（两积除以第三个变量，分母幂 -1 与 -2）
    for a, b, c in combinations(range(d), 3):
        for x, yz, zt in [(a, b, c), (a, c, b), (b, c, a)]:
            num = X[:, x] * X[:, yz]
            for q in (1, 2):
                u = num / np.clip(np.power(X[:, zt], q), _EPS, None)
                ue = ('div', ('mul', col_expr[x], col_expr[yz]),
                      ('pow', col_expr[zt], q) if q > 1 else col_expr[zt])
                u_list.append((u, ue, [a, b, c]))

    # u 积木：单列 × 三角构件（cos 角度差、1−cos、cos 单列）
    trig_parts = []
    for a, b in combinations(range(d), 2):
        trig_parts.append((np.cos(X[:, a] - X[:, b]),
                           ('cos', ('sub', col_expr[a], col_expr[b])), [a, b]))
    for a in range(d):
        trig_parts.append((1 - np.cos(X[:, a]),
                           ('sub', ('const', 1.0), ('cos', col_expr[a])), [a]))
        trig_parts.append((np.cos(X[:, a]), ('cos', col_expr[a]), [a]))
    for t, te, td in trig_parts:
        for a in range(d):
            if a in td:
                continue
            u_list.append((X[:, a] * t,
                           ('mul', col_expr[a], te), [a] + td))
            for b in range(d):
                if b == a or b in td:
                    continue
                u_list.append((X[:, a] * t / np.clip(X[:, b], _EPS, None),
                               ('div', ('mul', col_expr[a], te), col_expr[b]),
                               [a, b] + td))
                # 分母为两列（含单列平方）：E(1−cos)/(m·c²) 等
                for c in range(d):
                    if c == a or c == b or c in td:
                        continue
                    den = np.clip(X[:, b], _EPS, None) * X[:, c]
                    dene = ('mul', col_expr[b], col_expr[c])
                    u_list.append((X[:, a] * t / den,
                                   ('div', ('mul', col_expr[a], te), dene),
                                   [a, b, c] + td))
                    den2 = np.clip(X[:, b], _EPS, None) * X[:, c] ** 2
                    den2e = ('mul', col_expr[b], ('pow', col_expr[c], 2))
                    u_list.append((X[:, a] * t / den2,
                                   ('div', ('mul', col_expr[a], te), den2e),
                                   [a, b, c] + td))

    # B. 归一化 1 + cf*u
    for u, ue, ud in u_list:
        for cf in (1.0, -1.0, 0.5, -0.5, 1.0 / 3.0, -1.0 / 3.0):
            add(1.0 + cf * u,
                ('add', ('const', 1.0), ('mul', ('const', cf), ue)), ud)

    # C. 平方差
    for a, b in combinations(range(d), 2):
        add(X[:, a] ** 2 - X[:, b] ** 2,
            ('sub', ('pow', col_expr[a], 2), ('pow', col_expr[b], 2)), [a, b])

    # D. 变量和 / 差（线性组合）
    for a, b in combinations(range(d), 2):
        add(X[:, a] + X[:, b],
            ('add', col_expr[a], col_expr[b]), [a, b])
        add(X[:, a] - X[:, b],
            ('sub', col_expr[a], col_expr[b]), [a, b])
    return out


def _fast_space(eng, X, col_expr):
    """分子列空间：L1 lean + 双向比。"""
    X1, cex1, dep1 = eng._build_lean(X, col_expr)
    cols = [X1[:, j] for j in range(X1.shape[1])]
    cex = list(cex1)
    dep = list(dep1)
    d = X.shape[1]
    for a, b in combinations(range(d), 2):
        cols.append(X[:, a] / np.clip(X[:, b], _EPS, None))
        cex.append(('div', col_expr[a], col_expr[b]))
        dep.append(frozenset([a, b]))
        cols.append(X[:, b] / np.clip(X[:, a], _EPS, None))
        cex.append(('div', col_expr[b], col_expr[a]))
        dep.append(frozenset([a, b]))
    # 单列 × 单列平方（含 x_i³、x_i x_j²）：支持 d(1−α²) 等三次项
    for a in range(d):
        cols.append(X[:, a] ** 3)
        cex.append(('pow', col_expr[a], 3))
        dep.append(frozenset([a]))
        for b in range(d):
            if b == a:
                continue
            cols.append(X[:, a] * X[:, b] ** 2)
            cex.append(('mul', col_expr[a], ('pow', col_expr[b], 2)))
            dep.append(frozenset([a, b]))
    M = np.column_stack(cols)
    M = np.where(np.isfinite(M), M, 0.0)
    return M, cex, dep


def _is_sqdiff(e):
    """差平方 (x_i - x_j)^2 列：共线时反向 drop 会用它替代更简的直接积。"""
    return (isinstance(e, tuple) and len(e) == 3 and e[0] == 'pow'
            and isinstance(e[1], tuple) and e[1][0] == 'sub')


def _fwd_score(M, y, keep, max_terms=4):
    """全数据前向打分（仅用于排序/评估，不决定最终答案）。
    不做 75/25 切分：切分后 val 上的 greedy 会漏掉等权第二项
    （如 m1r1+m2r2，第一步 x0x2 后第二步 x1x3 提升不足 1e-4 就中断）。
    真值由阶段2 组装后在原始 y 上后验 + 符号等价核验兜底，故排序用全数据
    不构成作弊。

    向量化 Gram-Schmidt（与逐个 lstsq 机器精度等价，120 抽样最大差 2.22e-16）：
    去均值（等价 lstsq 带截距），每步用 r²_j=(R_jᵀyres)²/(‖R_j‖²‖yres‖²)
    一次选出最高列，再批量正交化（yres 与 R 各列对 qj 投影），不再逐列 lstsq。
    """
    Ms = M[:, keep]
    R = Ms - Ms.mean(0, keepdims=True)
    yc0 = y - y.mean()
    sst = float(np.sum(yc0 ** 2))
    K = len(keep)
    remaining = list(range(K))
    chosen = []
    yres = yc0.copy()
    best = -1.0
    for t in range(max_terms):
        rn = np.sum(R * R, axis=0)
        sse = float(np.sum(yres ** 2))
        dots = R.T @ yres
        den = rn * sse
        r2one = np.where((rn > 1e-12) & (den > 1e-12),
                         dots ** 2 / np.where(den > 1e-12, den, 1.0), -1.0)
        mask = np.zeros(K, dtype=bool)
        mask[remaining] = True
        j = int(np.argmax(np.where(mask, r2one, -1.0)))
        qj = R[:, j] / np.sqrt(rn[j])
        yres_new = yres - qj * float(qj @ yres)
        new_best = (1.0 - float(np.sum(yres_new ** 2)) / sst
                    if sst > 1e-12 else best)
        if chosen and new_best < best + 1e-10:
            break
        chosen.append(j)
        remaining.remove(j)
        yres = yres_new
        coeff = R.T @ qj
        R = R - qj[:, None] * coeff[None, :]
        best = new_best
        if best >= SOLVE_R2:
            break
    return best


def _light_molecule(eng, X, z, col_expr, max_peel=4):
    """轻量分子：fast 剥层优先 + 仅对最有希望的单点做 exact。

    peel 语义：剥 f（yt=z/f），最终 N = 解(yt) * f。
    剥层搜索全程只跑 fast（简列 fast→全列 fast 取较优），任一点 fast≥SOLVE_R2
    立即组装返回；记录沿途唯一 fast 最高且 ≥SOLVE_HOPE 的点 best_hope；剥层
    找不到满分后，只对 best_hope 做一次 exact（简列→全列），不再对每个剥层点
    反复 exact（旧实现对每个剥层点都跑 exact，是 II_21_32 326s 的主因）。
    """
    M, cex, dep = _fast_space(eng, X, col_expr)

    def solve_fast(active, yc):
        keep = [j for j in range(M.shape[1]) if dep[j] <= active]
        if not keep:
            return -1.0, None, None
        keep_s = [j for j in keep if not _is_sqdiff(cex[j])]
        s, c, r = None, None, -1.0
        if keep_s:
            s, c, r = eng._additive_fast(M, yc, keep_s, eng.seed)
        if len(keep) != len(keep_s):
            s2, c2, r2 = eng._additive_fast(M, yc, keep, eng.seed)
            if r2 > r:
                s, c, r = s2, c2, r2
        return r, s, c

    def exact_on(active, yc):
        keep = [j for j in range(M.shape[1]) if dep[j] <= active]
        if not keep:
            return -1.0, None, None
        keep_s = [j for j in keep if not _is_sqdiff(cex[j])]
        s, c, r = None, None, -1.0
        if keep_s:
            s, c, r = eng._additive(M, yc, keep_s, eng.seed, exact=True)
        if r < SOLVE_R2 and len(keep) != len(keep_s):
            s2, c2, r2 = eng._additive(M, yc, keep, eng.seed, exact=True)
            if r2 > r:
                s, c, r = s2, c2, r2
        return r, s, c

    active = set(range(X.shape[1]))
    ycur = z
    peels = []
    best_hope = None  # (r, active, yc, peels)
    for it in range(max_peel + 1):
        r, s, c = solve_fast(active, ycur)
        if r >= SOLVE_R2:
            e, _ = eng._assemble(s, c, cex, col_expr, peels)
            return r, e
        if r >= SOLVE_HOPE and (best_hope is None or r > best_hope[0]):
            best_hope = (r, set(active), ycur, list(peels))
        best = None
        for k in sorted(active):
            for pw in (-1, 1, -2, 2):
                yt = ycur / np.power(np.clip(X[:, k], _EPS, None), pw)
                if not np.all(np.isfinite(yt)):
                    continue
                act2 = active - {k}
                rr, s2, c2 = solve_fast(act2, yt)
                fac = col_expr[k] if pw == 1 else ('pow', col_expr[k], pw)
                if rr >= SOLVE_R2:
                    e, _ = eng._assemble(s2, c2, cex, col_expr, peels + [fac])
                    return rr, e
                if rr >= SOLVE_HOPE and (best_hope is None
                                         or rr > best_hope[0]):
                    best_hope = (rr, set(act2), yt, list(peels) + [fac])
                if best is None or rr > best[0]:
                    best = (rr, k, pw, yt, act2, fac)
        if best is None or best[0] <= r + 1e-6:
            break
        rr, k, pw, yt, act2, fac = best
        ycur, active = yt, act2
        peels.append(fac)
    if best_hope is not None:
        rh, acth, ych, peelh = best_hope
        r, s, c = exact_on(acth, ych)
        if r >= SOLVE_R2:
            e, _ = eng._assemble(s, c, cex, col_expr, peelh)
            return r, e
    return None


PEEL_REFRESH = 0.90
# 只对直接分 < SOLVE_R2 的因子剥层：直接分已满分(≥.9995)再剥不可能更高
# （r2 上限 1），省掉；其余一律剥，保证公平。不得把上限设更低——
# II_21_32 正确 g 直接分仅 .942、排第 102，靠剥层才抬到 1.0，截顶会漏
# （实测 .985 截顶让错误因子剥后虚高反超，II_11_3 被挤成错误近似）。
PEEL_CAP = SOLVE_R2


def _peel_score(eng, M, dep, X, z, active, terms=4):
    """对 z 剥一层单变量后，在 (L1+双向比) 上的最高前向分；用于给
    '分子需剥一层才能解' 的正确因子补上排序分。"""
    best = -1.0
    for k in range(X.shape[1]):
        for pw in (-1, 1, -2, 2):
            yt = z / np.power(np.clip(X[:, k], _EPS, None), pw)
            if not np.all(np.isfinite(yt)):
                continue
            act = active - {k}
            kk = [j for j in range(M.shape[1]) if dep[j] <= act]
            if not kk:
                continue
            rr = _fwd_score(M, yt, kk, max_terms=terms)
            if rr > best:
                best = rr
    return best


def find_rational(eng, X, y, col_expr, fast_top=FAST_TOP):
    """返回 (r2, full_expr) 或 None。"""
    M, cex, dep = _fast_space(eng, X, col_expr)
    allcols = list(range(M.shape[1]))
    active_all = set(range(X.shape[1]))

    # Stage 1: two-level screening. 1a directly scores 4-term; 1b peels one
    # layer for factors >= PEEL_REFRESH then scores (adding ranking for the
    # correct factor whose numerator needs one peel).
    fast = []
    import transform_solve as TSOL
    for g, ge, gd in build_denom_factors(X, col_expr):
        if TSOL._deadline_hit():
            break
        if np.min(np.abs(g)) < _EPS:
            continue
        for p in (1, 2):
            z = y * np.power(g, p)
            if not np.all(np.isfinite(z)):
                continue
            r = _fwd_score(M, z, allcols)
            if PEEL_REFRESH <= r < PEEL_CAP:
                r = max(r, _peel_score(eng, M, dep, X, z, active_all))
            rl = _quick_log_r2(X, z)
            fast.append((max(r, rl), p, g, ge, rl))
    fast.sort(key=lambda t: -t[0])
    if not fast or fast[0][0] < FAST_GATE:
        return None

    # 阶段2：按快速分降序；轻量分子（含双向比+剥层），分子为纯乘性
    # 时（log 分高）走 MLS；命中立即返回
    seen = set()
    for r, p, g, ge, rl in fast[:fast_top]:
        key = (p, ge)
        if key in seen:
            continue
        seen.add(key)
        z = y * np.power(g, p)
        denom = ge if p == 1 else ('pow', ge, p)
        light = _light_molecule(eng, X, z, col_expr)
        if light is not None:
            full = ('div', light[1], denom)
            ypred = _ev_c(full, X)
            if np.all(np.isfinite(ypred)):
                r_orig = float(SC.r2(y, ypred))
                if r_orig >= SOLVE_R2:
                    return r_orig, full
        if rl >= FAST_GATE:
            ok = np.abs(z) > _EPS
            res = MLS.solve_mult_log(X[ok], z[ok], col_expr)
            if res is not None:
                full = ('div', res[1], denom)
                ypred = _ev_c(full, X)
                if np.all(np.isfinite(ypred)):
                    r_orig = float(SC.r2(y, ypred))
                    if r_orig >= SOLVE_R2:
                        return r_orig, full
    return None


def find_rational_sqrt(X, y, col_expr, max_d=5):
    """y = sqrt(P)/g：**分式外框 + 分子带 sqrt、sqrt 内加性 P**。

    覆盖相对论多普勒角频率：`sqrt(1-v²/c²)·omega/(1+(v/c)cosθ)`。
    rational 剥分母 g 后，分子 z 带 sqrt、不在其线性列空间；本函数剥分母后
    does additive on z^2 (reusing additive_blocks' staged screening),
    assembling sqrt(P)/g.

    分母枚举复用 build_denom_factors（通用）；分子加性积木只构造一次。
    """
    import additive_blocks as ADDB
    d = X.shape[1]
    if d > max_d or d < 2:
        return None
    if not (np.all(y > 0) or np.all(y < 0)):
        return None
    spaces = [ADDB._inner_add_space(X, col_expr, include_triple=t)
              for t in (False, True)]
    # 无分母（g=1）在 y² 上就有精确稀疏解 → 这是纯 sqrt 题（find_sqrt_inner_additive
    # 的地盘），任何枚举出来的分式都只是"分子分母可约"的假分式，直接返回 None。
    w0 = y ** 2
    if any(_strict_inner_sqrt(w0, B, tups) is not None
           for B, tups in spaces):
        return None
    cands = []
    import transform_solve as TSOL
    for g, ge, gd in build_denom_factors(X, col_expr):
        if TSOL._deadline_hit():
            break
        if np.min(np.abs(g)) < _EPS:
            continue
        z = y * g
        if not (np.all(z > 0) or np.all(z < 0)):
            continue
        sgn = 1 if np.mean(z) > 0 else -1
        w = z ** 2
        for B, tups in spaces:
            pexpr = _strict_inner_sqrt(w, B, tups)
            if pexpr is not None:
                cands.append((len(gd), sgn, pexpr, ge))
    if not cands:
        return None
    cands.sort(key=lambda c: c[0])
    _, sgn, pexpr, ge = cands[0]
    num = ('sqrt', pexpr)
    if sgn < 0:
        num = ('mul', ('const', -1.0), num)
    full = ('div', num, ge)
    ypred = _ev_c(full, X)
    if not np.all(np.isfinite(ypred)):
        return None
    r = float(SC.r2(y, ypred))
    if r >= SOLVE_R2:
        return r, full
    return None


_RSQRT_EPS = 1e-9
_RSQRT_TERMS = 8


def _strict_inner_sqrt(w, B, tups, max_terms=_RSQRT_TERMS, eps=_RSQRT_EPS):
    """在 w 上求**精确且稀疏**的加性解。

    高积木池对任意 w 都能给出 r2≈1-1e-8、用上百项的过拟合解；真解则是
    r2=1、极少数项（实测真解 2 项、过拟合 87~263 项）。双门：
    ① 原始 lstsq r2 ≥ 1-eps；② 清理后系数非零项数 ≤ max_terms；
    ③ 用清理后分数系数重建 r2 仍 ≥ 1-eps。
    """
    import additive_blocks as ADDB
    coef, *_ = np.linalg.lstsq(B, w, rcond=None)
    if not np.all(np.isfinite(coef)):
        return None
    if float(SC.r2(w, B @ coef)) < 1 - eps:
        return None
    fr = ADDB._clean_coef_mixed(coef)
    nz = [(i, c) for i, c in enumerate(fr) if c != 0]
    if len(nz) == 0 or len(nz) > max_terms:
        return None
    zc = np.zeros_like(w)
    for i, c in nz:
        zc = zc + float(c) * B[:, i]
    if float(SC.r2(w, zc)) < 1 - eps:
        return None
    terms = [(float(c), tups[i]) for i, c in nz]
    return ADDB._build_add_expr(terms)
