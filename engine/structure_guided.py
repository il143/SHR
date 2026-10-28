# -*- coding: utf-8 -*-
"""Structure-guided general solver (self-contained, no per-problem patches).

Inner core: the column set is enabled in stages -- small base (original /
  powers / first ratios / sin2-cos2) -> compound g2 (diff/sum/squared-diff)
  -> pair products -> power ratios / linear fractions -> compound 1/g2;
  each stage first greedily takes new columns, then crosses selected x stage,
  stopping at r2=1; a multivariable monomial MLS is the fallback.
  The small base goes first so redundant columns do not mislead greedy.
Outer: identify and peel wrappers -- additive terms, unary sqrt,
  multiplicative factors (probes fix the power/cos, else enumerate single
  variables and binary power ratios, each cored once), shifted cos(diff);
  the multivariable multiplicative sweep is costliest and tried last; peel
  until a strict r2=1 core is exposed, then reassemble in reverse order.
Probes only narrow the space and do not act as a start gate; the final
computation is strict r2=1. The exactness check lives outside this module
(the fit does not read ground truth).
"""
import itertools
import numpy as np

import mult_log_solver as MLS
import structure_tools as ST
from shr_surface import Surface
from auto_decompose_v5 import _ev_c

GATE = 1 - 1e-7
MAX_TERMS = 8
MAX_DEPTH = 6
TRANS_DEPTH = 5
CSYM_TOL = 0.15


def _var(i):
    return ("c", i)


_NODE_KEY_CACHE = {}


def _node_key(e):
    if isinstance(e, tuple):
        ck = _NODE_KEY_CACHE.get(e)
        if ck is not None:
            return ck
        ck = (e[0],) + tuple(_node_key(a) for a in e[1:])
        _NODE_KEY_CACHE[e] = ck
        return ck
    return e


def _dedup(cols):
    seen, out = set(), []
    for v, nd in cols:
        key = _node_key(nd)
        if key not in seen and np.all(np.isfinite(v)) and np.std(v) > 1e-12:
            seen.add(key)
            out.append((np.asarray(v, float), nd))
    return out


# ---------------- 分层列集（只依赖 X，按 X 缓存） ----------------

_CACHE = {}


def _xkey(X):
    return X.shape, round(float(X.sum()), 3)


def small_base(X, feats):
    d = X.shape[1]
    cols = []
    for i in range(d):
        v = X[:, i]
        cols.append((v, _var(i)))
        for p in (-3, -2, -1, 2, 3):
            if p < 0:
                if np.min(np.abs(v)) > 1e-6:
                    cols.append((np.abs(v) ** p, ("pow", _var(i), p)))
            else:
                cols.append((v ** p, ("pow", _var(i), p)))
    for a in range(d):
        for b in range(d):
            if a != b and np.min(np.abs(X[:, b])) > 1e-6:
                cols.append((X[:, a] / X[:, b], ("div", _var(a), _var(b))))
    for i in range(d):
        cols.append((np.sin(X[:, i]) ** 2, ("pow", ("sin", _var(i)), 2)))
        cols.append((np.cos(X[:, i]) ** 2, ("pow", ("cos", _var(i)), 2)))
    return _dedup(cols)


def _compound_raw(X, feats):
    d = X.shape[1]
    cols = []
    for a in range(d):
        for b in range(a + 1, d):
            va, vb = X[:, a], X[:, b]
            cols.append((va - vb, ("sub", _var(a), _var(b))))
            cols.append((va + vb, ("add", _var(a), _var(b))))
            cols.append((va ** 2 - vb ** 2,
                         ("sub", ("pow", _var(a), 2), ("pow", _var(b), 2))))
    for i in range(d):
        for j in range(d):
            for k in range(j + 1, d):
                if i != j and i != k:
                    cols.append((X[:, i] - X[:, j] * X[:, k],
                                 ("sub", _var(i),
                                  ("mul", _var(j), _var(k)))))
    return _dedup(cols)


def pair_cols(X):
    d = X.shape[1]
    cols = []
    for a in range(d):
        for b in range(a + 1, d):
            cols.append((X[:, a] * X[:, b], ("mul", _var(a), _var(b))))
    return _dedup(cols)


def power_ratio_cols(X):
    """幂比 X_i^p / X_j^q（p∈1..3, q∈1..2）。"""
    d = X.shape[1]
    cols = []
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            for p in (1, 2, 3):
                for q in (1, 2):
                    if np.min(np.abs(X[:, j])) > 1e-6:
                        cols.append((X[:, i] ** p / np.abs(X[:, j]) ** q,
                                     ("div", ("pow", _var(i), p),
                                      ("pow", _var(j), q))))
    return _dedup(cols)


def linear_frac_cols(X):
    """线性分式 (v-c)/(v+d)，c∈0..3, d∈1..3。"""
    d = X.shape[1]
    cols = []
    for i in range(d):
        v = X[:, i]
        for c in range(0, 4):
            for dd in range(1, 4):
                cols.append(((v - c) / (v + dd),
                             ("div", ("add", _var(i), ("const", float(-c))),
                              ("add", _var(i), ("const", float(dd))))))
    return _dedup(cols)


def get_pools(X, feats):
    key = _xkey(X)
    if key in _CACHE:
        return _CACHE[key]
    L0 = small_base(X, feats)
    comp = _compound_raw(X, feats)
    L1 = [(g ** 2, ("pow", nd, 2)) for g, nd in comp]
    L2 = pair_cols(X)
    L3 = power_ratio_cols(X)
    L4 = linear_frac_cols(X)
    L5 = [(1.0 / g ** 2, ("div", ("const", 1.0), ("pow", nd, 2)))
          for g, nd in comp if np.min(np.abs(g)) > 1e-3]
    layers = [_dedup(L) for L in (L0, L1, L2, L3, L4, L5)]
    layers = [L for L in layers if L]
    pools = {"layers": layers}
    _CACHE[key] = pools
    return pools


# ---------------- 单步前向（指定层，可选已选×层 cross） ----------------

def _linear_expr(cf, nodes):
    terms = []
    for k in range(1, len(cf)):
        c = float(cf[k])
        if abs(c) < 1e-7:
            continue
        if abs(c - 1) < 1e-7:
            terms.append(nodes[k - 1])
        elif abs(c + 1) < 1e-7:
            terms.append(("mul", ("const", -1.0), nodes[k - 1]))
        else:
            terms.append(("mul", ("const", c), nodes[k - 1]))
    if abs(float(cf[0])) >= 1e-7:
        terms.append(("const", float(cf[0])))
    if not terms:
        return ("const", 0.0)
    e = terms[0]
    for t in terms[1:]:
        e = ("add", e, t)
    return e


def _fwd_step(y, lv, ln, Xm, chosen, cross_v=None, cross_n=None):
    ss = np.sum((y - y.mean()) ** 2)
    beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    resid = y - Xm @ beta
    pv, pn = [], []
    for v, nd in zip(lv, ln):
        if _node_key(nd) not in chosen:
            pv.append(v); pn.append(nd)
    if cross_v is not None:
        for sv, sn in zip(cross_v, cross_n):
            for v, nd in zip(lv, ln):
                e = ("mul", sn, nd)
                if _node_key(e) not in chosen:
                    pv.append(sv * v); pn.append(e)
    if not pv:
        return None
    C = np.column_stack(pv)
    ac, *_ = np.linalg.lstsq(Xm, C, rcond=None)
    Cp = C - Xm @ ac
    cpn = np.sum(Cp ** 2, axis=0)
    rr = resid @ Cp
    delta = np.zeros(len(pv))
    good = cpn > 1e-12
    delta[good] = rr[good] ** 2 / (cpn[good] * ss)
    jb = int(np.argmax(delta))
    if delta[jb] <= 1e-12:
        return None
    return pv[jb], pn[jb]


def _add_col(sel_v, sel_n, chosen, Xm, v, nd):
    sel_v.append(v); sel_n.append(nd); chosen.add(_node_key(nd))
    return np.column_stack([Xm, v])


def super_pool(X, feats):
    """把所有列层（L0–L5）flatten 成一个去重候选池，让正确列在每步都能比较。"""
    key = _xkey(X)
    if key in _POOL_CACHE:
        return _POOL_CACHE[key]
    pools = get_pools(X, feats)
    out, seen = [], set()
    for layer in pools["layers"]:
        for v, nd in layer:
            k = _node_key(nd)
            if k not in seen:
                seen.add(k)
                out.append((v, nd))
    _POOL_CACHE[key] = out
    return out


_POOL_CACHE = {}


def greedy_fwd(X, y, feats, max_terms=MAX_TERMS, verbose=False):
    """全列 greedy：每步候选 = 全列 + 已选×全列，选 r2 增益最大，带截距。"""
    pool = super_pool(X, feats)
    lv = [c[0] for c in pool]
    ln = [c[1] for c in pool]
    n = len(y)
    Xm = np.ones((n, 1))
    sel_v, sel_n, chosen = [], [], set()

    def r2cf():
        cf, *_ = np.linalg.lstsq(Xm, y, rcond=None)
        return 1 - np.sum((y - Xm @ cf) ** 2) / np.sum(
            (y - y.mean()) ** 2), cf

    for step in range(max_terms):
        st = _fwd_step(y, lv, ln, Xm, chosen,
                       cross_v=sel_v, cross_n=sel_n)
        if st is None:
            break
        v, nd = st
        Xm = _add_col(sel_v, sel_n, chosen, Xm, v, nd)
        r2, cf = r2cf()
        if verbose:
            print("    pick %s -> r2=%.6f" % (nd, r2))
        if r2 >= GATE:
            return r2, _linear_expr(cf, sel_n)
    r2, cf = r2cf()
    return r2, _linear_expr(cf, sel_n)


# ---------------- 内层 core：分层前向 + 多变量单项式 ----------------

def core_solve(Xs, ys, feats, pools=None):
    best = None
    r2, node = greedy_fwd(Xs, ys, feats)
    if r2 >= GATE:
        best = (r2, node)
    ce = [("c", k) for k in range(Xs.shape[1])]
    msol = MLS.solve_mult_log(Xs, ys, ce)
    if msol is not None and msol[0] >= GATE:
        if best is None or _complexity(msol[1]) < _complexity(best[1]):
            best = (msol[0], msol[1])
    if best is not None:
        return best
    return r2, None


def _complexity(node):
    if not isinstance(node, tuple):
        return 1
    if node[0] == "add":
        return _complexity(node[1]) + _complexity(node[2])
    return 1 + sum(_complexity(a) for a in node[1:] if isinstance(a, tuple))


# ---------------- 结构探针：单变量乘性幂次 / cos ----------------

def _state_surface(X, y, epochs=4000):
    return Surface(X, y, epochs=epochs)


def outer_power_probe(S, X, feats):
    """多基准点扫描、投票定最外层乘性幂次（宽容差）或 cos（符号由系数吸收）。"""
    d = X.shape[1]
    lo, hi = X.min(0), X.max(0)
    rng = np.random.default_rng(7)
    bases = [(lo + hi) / 2.0]
    for _ in range(5):
        bases.append(X[rng.integers(len(X))].copy())
    nodes, vals = [], []
    for i in range(d):
        g0 = lo[i] + 0.1 * (hi[i] - lo[i])
        g1 = hi[i] - 0.1 * (hi[i] - lo[i])
        grid = np.linspace(g0, g1, 100)
        r2cs, phis, slopes, flips = [], [], [], 0
        for b in bases:
            P = np.tile(b, (100, 1)); P[:, i] = grid
            curve = S.query(P)
            Mc = np.column_stack([np.cos(grid), np.sin(grid)])
            cc, *_ = np.linalg.lstsq(Mc, curve, rcond=None)
            cfit = Mc @ cc
            ss = np.sum((curve - curve.mean()) ** 2)
            r2c = 1 - np.sum((curve - cfit) ** 2) / ss if ss > 0 else 0
            r2cs.append(r2c)
            phis.append(np.arctan2(cc[1], cc[0]))
            if curve.min() < 0 < curve.max():
                flips += 1
            if (np.all(curve > 0) or np.all(curve < 0)) and np.all(
                    np.abs(grid) > 1e-6):
                A = np.column_stack([np.ones(100), np.log(np.abs(grid))])
                cf, *_ = np.linalg.lstsq(A, np.log(np.abs(curve)), rcond=None)
                lc = np.log(np.abs(curve))
                ss2 = np.sum((lc - lc.mean()) ** 2)
                r2l = 1 - np.sum((lc - A @ cf) ** 2) / ss2
                if r2l >= 0.99:
                    slopes.append(cf[1])
        r2m = float(np.median(r2cs))
        phimod = float(np.median(
            [abs(((p + np.pi / 2) % np.pi) - np.pi / 2) for p in phis]))
        if r2m >= 0.98 and phimod <= 0.12 and flips >= 3:
            nodes.append(("cos", _var(i)))
            vals.append(np.cos(X[:, i]))
            continue
        if len(slopes) >= 2:
            sm = float(np.median(slopes))
            p = int(round(sm))
            near = sum(1 for s in slopes if abs(s - p) < 0.3)
            if (p != 0 and abs(sm - p) < 0.25
                    and near >= max(2, (len(slopes) + 1) // 2)):
                nodes.append(_var(i) if p == 1 else ("pow", _var(i), p))
                vals.append(X[:, i] ** p)
    if not nodes:
        return None, None
    Cnode, Cval = nodes[0], vals[0]
    for nd, vv in zip(nodes[1:], vals[1:]):
        Cnode = ("mul", Cnode, nd)
        Cval = Cval * vv
    return Cnode, Cval


# ---------------- 枚举单变量/二元幂比（剥后只 core 一次） ----------------

def single_power_factors(X):
    d = X.shape[1]
    out = []
    for i in range(d):
        for p in (-2, -1, 1, 2):
            f = X[:, i] if p == 1 else X[:, i] ** p
            if np.min(np.abs(f)) > 1e-6:
                fn = _var(i) if p == 1 else ("pow", _var(i), p)
                out.append((f, fn))
    return out


def pair_power_factors(X):
    d = X.shape[1]
    out = []
    for a in range(d):
        for b in range(a + 1, d):
            for p in (1, 2):
                for q in (1, 2):
                    if np.min(np.abs(X[:, b])) > 1e-6:
                        f = X[:, a] ** p / np.abs(X[:, b]) ** q
                        fn = ("div", ("pow", _var(a), p),
                              ("pow", _var(b), q))
                        out.append((f, fn))
    return out


# ---------------- 平移剥层 dfs（sub1 / divcos / sqrt） ----------------

def peel_translation_dfs(S, X, y, feats, i, j, dval, pools,
                         max_depth=TRANS_DEPTH):
    cosd = np.cos(dval)
    results = []

    def acts(s, msk):
        out = [("sub1", s - 1.0, msk)]
        nm = msk & (np.abs(cosd) > 0.12)
        if nm.sum() > 30:
            out.append(("divcos", s / cosd, nm))
        if np.all(s[msk] > 0) or np.all(s[msk] < 0):
            out.append(("sqrt", s ** 2, msk))
        return out

    def inv(kind, node):
        if kind == "sub1":
            return ("add", node, ("const", 1.0))
        if kind == "divcos":
            return ("mul", node, ("cos", ("sub", _var(i), _var(j))))
        return ("sqrt", node)

    seen = set()

    def rec(s, msk, path, depth):
        r2, node = core_solve(X[msk], s[msk], feats)
        if r2 >= GATE and node is not None:
            for k in path[::-1]:
                node = inv(k, node)
            results.append((len(path), node))
            return
        if depth >= max_depth:
            return
        for kind, sp, nm in acts(s, msk):
            key = (tuple(path + [kind]), round(float(np.mean(sp[nm])), 5))
            if key in seen or not np.all(np.isfinite(sp[nm])):
                continue
            seen.add(key)
            rec(sp, nm, path + [kind], depth + 1)

    rec(y, np.ones(len(y), bool), [], 0)
    if not results:
        return None
    results.sort(key=lambda t: t[0])
    return results[0]


# ---------------- 多变量乘性 sweep（宽容差定符号，最贵最后） ----------------

def sweep_Csym(S, X, y, feats, i, j):
    d = X.shape[1]
    lo_i, hi_i = X[:, i].min(), X[:, i].max()

    def safe_R(xjv):
        return min(abs((lo_i - xjv) * 0.35), abs((hi_i - xjv) * 0.35), np.pi)

    other = [k for k in range(d) if k not in (i, j)]
    if not other:
        return None, None
    rng = np.random.default_rng(3)
    Xo, Cv = [], []
    for _ in range(100):
        b = X[rng.integers(len(X))].copy()
        R = safe_R(b[j])
        if R < 0.4:
            continue
        grid = np.linspace(-R, R, 120)
        P = np.tile(b, (120, 1))
        P[:, i] = b[j] + grid
        curve = S.query(P)
        M = np.column_stack([np.ones(120), np.cos(grid)])
        coef, *_ = np.linalg.lstsq(M, curve, rcond=None)
        Xo.append([b[k] for k in other])
        Cv.append(float(coef[0]))
    Xo = np.array(Xo); Cv = np.array(Cv)
    okp = np.abs(Cv) > 1e-9
    Xo, Cv = Xo[okp], Cv[okp]
    if len(Cv) < 20:
        return None, None
    logX, logC = np.log(np.abs(Xo)), np.log(np.abs(Cv))
    keep = np.ones(len(Cv), bool)
    cc = None
    for _ in range(4):
        A = np.column_stack([np.ones(keep.sum()), logX[keep]])
        cc, *_ = np.linalg.lstsq(A, logC[keep], rcond=None)
        resid = logC - (np.column_stack([np.ones(len(Cv)), logX]) @ cc)
        sd = np.std(resid[keep])
        bad = np.where(keep & (np.abs(resid) > 2.5 * sd))[0]
        if len(bad) == 0:
            break
        keep[bad] = False
    powers, node, used = {}, None, False
    for k in range(len(other)):
        raw = cc[k + 1]
        if abs(raw - round(raw)) <= CSYM_TOL:
            powers[other[k]] = int(round(raw))
    for col, p in powers.items():
        if p == 0:
            continue
        used = True
        t = _var(col) if p == 1 else (
            ("pow", _var(col), p) if p > 1 else
            ("div", ("const", 1.0), _var(col)) if p == -1 else
            ("div", ("const", 1.0), ("pow", _var(col), -p)))
        node = t if node is None else ("mul", node, t)
    if not used:
        return None, None
    val = np.ones(len(X))
    for col, p in powers.items():
        if p != 0:
            val = val * X[:, col] ** p
    return node, val


# ---------------- 外层（受控、有界；S 延迟建） ----------------

def _add_cands(X):
    d = X.shape[1]
    cands = [("const", 1.0)]
    for i in range(d):
        cands.append(_var(i))
    for a in range(d):
        for b in range(a + 1, d):
            cands.append(("mul", _var(a), _var(b)))
    return cands


def _after_add(X, z, feats, pools):
    """剥加性后有界验证：core(z)，或 z 全正剥 sqrt（z² core / 除单因子 core）。"""
    r2, zn = core_solve(X, z, feats, pools)
    if r2 >= GATE and zn is not None:
        return zn
    if np.all(z > 0) or np.all(z < 0):
        z2 = z ** 2
        r2q, qn = core_solve(X, z2, feats, pools)
        if r2q >= GATE and qn is not None:
            return ("sqrt", qn)
        for f, fn in single_power_factors(X):
            r2r, rn = core_solve(X, z2 / f, feats, pools)
            if r2r >= GATE and rn is not None:
                return ("sqrt", ("mul", fn, rn))
    return None


def _single_path(X, z, feats, pools):
    """core(z)、core(z²) 已失败后，试 z² 除单因子（返回 sqrt 节点）。"""
    z2 = z ** 2
    for f, fn in single_power_factors(X):
        r2r, rn = core_solve(X, z2 / f, feats, pools)
        if r2r >= GATE and rn is not None:
            return ("sqrt", ("mul", fn, rn))
    return None


def try_additive(X, y, feats, pools):
    """Two-stage screening: first all candidates core(z)/core(z^2), then zpos
    tries single factors in descending core(z^2) r2.

    Mechanism: the correct additive term's core(z^2) has the highest numerical
    r2 (anchor), so single factors are tried on it first;
    错误候选的 24 单因子大多是无效开销，按 r2 排序后靠后兜底，不改变结果。
    """
    entries = []
    for term in _add_cands(X):
        z = y - _ev_c(term, X)
        if not np.all(np.isfinite(z)):
            continue
        r2, zn = core_solve(X, z, feats, pools)
        if r2 >= GATE and zn is not None:
            return ("add", zn, term)
        zpos = np.all(z > 0) or np.all(z < 0)
        r2z2 = -1.0
        if zpos:
            r2q, qn = core_solve(X, z ** 2, feats, pools)
            if r2q >= GATE and qn is not None:
                return ("add", ("sqrt", qn), term)
            r2z2 = r2q
        entries.append((term, z, zpos, r2z2))
    zlist = [e for e in entries if e[2]]
    zlist.sort(key=lambda e: -e[3])
    TOP = 5
    for term, z, _, _ in zlist[:TOP] + zlist[TOP:]:
        inner = _single_path(X, z, feats, pools)
        if inner is not None:
            return ("add", inner, term)
    return None


def peel(X, y, feats, pools, S=None, depth=0, peeled_sqrt=False, swept=False):
    # core 验收：r2=1 且过符号门（真解）才接受；过拟合 r2=1（gate=False）不提前返回
    r2, node = core_solve(X, y, feats, pools)
    if r2 >= GATE and node is not None and symbol_gate(node):
        return r2, node
    if depth >= 4:
        return None
    # 加性（有界验证，顶层先试；不建 S）
    if depth == 0:
        ae = try_additive(X, y, feats, pools)
        if ae is not None:
            return GATE, ae
    # 一元 sqrt（全正）
    if not peeled_sqrt and (np.all(y > 0) or np.all(y < 0)):
        r = peel(X, y ** 2, feats, pools, S, depth + 1, True, swept)
        if r is not None:
            return r[0], ("sqrt", r[1])
    # 以下需结构识别（延迟建 S）
    if S is None:
        S = _state_surface(X, y)
    if depth == 0:
        found = ST.detect(S, X, hard_only=True)
        diffs = [r for r in found if r["comp"] == "diff"]
        # 乘性 power probe（一次定干净组合）
        cnode, cval = outer_power_probe(S, X, feats)
        if cnode is not None:
            r = peel(X, y / cval, feats, pools, S, depth + 1, peeled_sqrt,
                     swept)
            if r is not None:
                whole = ("mul", cnode, r[1])
                if symbol_gate(whole):
                    return r[0], whole
        # 平移剥层 dfs（原始 y）
        for rinfo in diffs:
            i, j = rinfo["i"], rinfo["j"]
            got = peel_translation_dfs(S, X, y, feats, i, j,
                                      X[:, i] - X[:, j], pools)
            if got is not None:
                return GATE, got[1]
        # 多变量乘性 sweep（最贵最后）；剥 C 后对残量再走平移 dfs
        if not swept and diffs:
            i, j = diffs[0]["i"], diffs[0]["j"]
            cnode, cval = sweep_Csym(S, X, y, feats, i, j)
            if cnode is not None:
                y1 = y / cval
                r = peel(X, y1, feats, pools, S, depth + 1, peeled_sqrt,
                         True)
                if r is not None:
                    return r[0], ("mul", cnode, r[1])
                got = peel_translation_dfs(S, X, y1, feats, i, j,
                                          X[:, i] - X[:, j], pools)
                if got is not None:
                    return GATE, ("mul", cnode, got[1])
    # 乘性枚举（单变量 / pair 幂比，纯数据 core）：收集 r2=1 候选，符号门优先、选最简
    hits = []
    for f, fn in single_power_factors(X):
        r2r, rn = core_solve(X, y / f, feats, pools)
        if r2r >= GATE and rn is not None:
            hits.append(("mul", fn, rn))
    for f, fn in pair_power_factors(X):
        r2r, rn = core_solve(X, y / f, feats, pools)
        if r2r >= GATE and rn is not None:
            hits.append(("mul", fn, rn))
    if hits:
        gated = [h for h in hits if symbol_gate(h)]
        chosen = gated if gated else hits
        chosen.sort(key=_complexity)
        return GATE, chosen[0]
    return None


# ================= 结构路径通用辅助 =================

def _r2(yt, p):
    return 1 - np.sum((yt - p) ** 2) / max(np.sum((yt - yt.mean()) ** 2), 1e-12)


def lin_corr(a, b):
    c1, c2 = a - a.mean(), b - b.mean()
    d = np.sqrt(np.dot(c1, c1) * np.dot(c2, c2))
    return np.dot(c1, c2) / d if d > 1e-12 else 0.0


def clean_base(X):
    """干净 base：原始 + 一元幂（不含比值），返回 (值, 节点)。"""
    out = [(X[:, i], _var(i)) for i in range(X.shape[1])]
    for i in range(X.shape[1]):
        for p in (-3, -2, -1, 2, 3):
            v = X[:, i]
            if p < 0:
                if np.min(np.abs(v)) > 1e-6:
                    out.append((np.abs(v) ** p, ("pow", _var(i), p)))
            else:
                out.append((v ** p, ("pow", _var(i), p)))
    return out


def find_big_term(r, X):
    """单项最大（与 r 相关性最强）；候选=干净 base + pair 乘积。"""
    base = clean_base(X)
    for a in range(X.shape[1]):
        for b in range(a + 1, X.shape[1]):
            base.append((X[:, a] * X[:, b], ("mul", _var(a), _var(b))))
    best = None
    for v, nd in base:
        c = lin_corr(r, v)
        if best is None or abs(c) > abs(best[0]):
            k = np.dot(r, v) / np.dot(v, v)
            best = (c, v, nd, k)
    return best[1], best[2], best[3]


def combo_fwd(r, X, comp, i, j, big_val, big_node):
    """Given composite 1/c^2 with the large term subtracted, a single combo
    anchors forward (no free base, no intercept)."""
    cval = ST._combo_value(comp, X, i, j)
    if np.min(np.abs(cval)) <= 1e-4:
        return None
    ig2 = 1.0 / cval ** 2
    cnode = ("pow", ST._combo_expr(comp, i, j), -2)
    base = clean_base(X)
    pool = [(ig2, cnode)]
    for v, nd in base:
        pool.append((ig2 * v, ("mul", cnode, nd)))
    sel_v, sel_n, used = [big_val], [big_node], set()
    for _ in range(8):
        best_r2, best_k = -1, -1
        for k in range(len(pool)):
            if k in used:
                continue
            A = np.column_stack(sel_v + [pool[k][0]])
            cf, *_ = np.linalg.lstsq(A, r, rcond=None)
            r2 = _r2(r, A @ cf)
            if r2 > best_r2:
                best_r2, best_k = r2, k
        if best_k < 0 or best_r2 < 0:
            break
        used.add(best_k)
        nv, nn = pool[best_k]
        sel_v.append(nv)
        sel_n.append(nn)
        for v, nd in base:
            pool.append((nv * v, ("mul", nn, nd)))
        if best_r2 >= GATE:
            break
    A = np.column_stack(sel_v)
    cf, *_ = np.linalg.lstsq(A, r, rcond=None)
    nterms = int(np.sum(np.abs(cf) > 1e-7))
    terms = []
    for k in range(len(cf)):
        c = float(cf[k])
        if abs(c) < 1e-7:
            continue
        if abs(c - 1) < 1e-7:
            terms.append(sel_n[k])
        elif abs(c + 1) < 1e-7:
            terms.append(("mul", ("const", -1), sel_n[k]))
        else:
            terms.append(("mul", ("const", c), sel_n[k]))
    expr = None
    if terms:
        expr = terms[0]
        for t in terms[1:]:
            expr = ("add", expr, t)
    return best_r2, nterms, expr


def flatten_mul(node):
    if isinstance(node, tuple) and node[0] == "mul":
        return flatten_mul(node[1]) + flatten_mul(node[2])
    return [node]


def _mul_from_factors(f):
    node = f[0]
    for x in f[1:]:
        node = ("mul", node, x)
    return node


# ---- Structure path 1: multiplicative large/small term family (outer subset
# enumeration + single combo anchor) ----

def _mult_combo_path(X, y, feats, S=None):
    if S is None:
        S = _state_surface(X, y)
    pnode, _ = outer_power_probe(S, X, feats)
    if pnode is None:
        return None
    factors = flatten_mul(pnode)
    d = X.shape[1]
    best = None
    for r in range(1, len(factors) + 1):
        for comb in itertools.combinations(range(len(factors)), r):
            onode = _mul_from_factors([factors[k] for k in comb])
            oval = _ev_c(onode, X)
            if oval is None or np.min(np.abs(oval)) < 1e-9:
                continue
            rr = y / oval
            bv, bn, bk = find_big_term(rr, X)
            z0 = rr - bv * bk
            rows = []
            for a in range(d):
                for b in range(a + 1, d):
                    for comp in ("diff", "ratio", "sqdiff"):
                        cval = ST._combo_value(comp, X, a, b)
                        if np.min(np.abs(cval)) <= 1e-4 or \
                                not np.all(np.isfinite(cval)):
                            continue
                        rows.append((abs(lin_corr(z0, 1.0 / cval ** 2)),
                                     a, b, comp))
            rows.sort(key=lambda t: -t[0])
            pairs, seen = [], set()
            for ac, a, b, comp in rows:
                if (a, b) not in seen:
                    seen.add((a, b))
                    pairs.append((a, b))
            for (a, b) in pairs[:3]:
                for comp in ("diff", "ratio", "sqdiff"):
                    res = combo_fwd(rr, X, comp, a, b, bv, bn)
                    if res is None:
                        continue
                    r2, nt, ex = res
                    if r2 >= GATE and ex is not None and \
                            (best is None or nt < best[0]):
                        best = (nt, ("mul", onode, ex))
    return None if best is None else best[1]


# ---- 结构路径 2：平移族（先 sweep 剥 C，再 dfs，mult_log 终止） ----

def _dfs_mult(X, r, i, j, dval, other, max_depth=5):
    Xbase = np.column_stack([X[:, k] for k in other])
    ce_base = [("c", k) for k in other]  # 全局变量索引
    cosd = np.cos(dval)
    results = []

    def actions(s, msk):
        out = [("sub1", s - 1.0, msk)]
        nm = msk & (np.abs(cosd) > 0.12)
        if nm.sum() > 30:
            out.append(("divcos", s / cosd, nm))
        if np.all(s[msk] > 0) or np.all(s[msk] < 0):
            out.append(("sqrt", s ** 2, msk))
        return out

    def inverse(kind, node):
        if kind == "sub1":
            return ("add", node, ("const", 1.0))
        if kind == "divcos":
            return ("mul", node, ("cos", ("sub", _var(i), _var(j))))
        return ("sqrt", node)

    seen = set()

    def rec(s, msk, path, depth):
        key = (tuple(path), round(float(np.mean(s[msk])), 4))
        if key in seen:
            return
        seen.add(key)
        res = MLS.solve_mult_log(Xbase[msk], s[msk], ce_base)
        if res is not None and res[0] >= GATE:
            node = res[1]
            for k in path[::-1]:
                node = inverse(k, node)
            results.append((len(path), node))
            return
        if depth >= max_depth:
            return
        for kind, sp, nm in actions(s, msk):
            if np.all(np.isfinite(sp[nm])):
                rec(sp, nm, path + [kind], depth + 1)

    rec(r, np.ones(len(r), bool), [], 0)
    if not results:
        return None
    results.sort(key=lambda t: t[0])
    return results[0][1]


def _translation_path(X, y, feats, S=None):
    if S is None:
        S = _state_surface(X, y)
    found = ST.detect(S, X, hard_only=True, verbose=False)
    diffs = [(r["i"], r["j"], r["conf"]) for r in found
             if r["comp"] == "diff"]
    if not diffs:
        return None
    i, j, _ = max(diffs, key=lambda t: t[2])
    dval = X[:, i] - X[:, j]
    other = [k for k in range(X.shape[1]) if k not in (i, j)]
    cnode, cval = sweep_Csym(S, X, y, feats, i, j)
    if cnode is None:
        return _dfs_mult(X, y, i, j, dval, other)
    inner = _dfs_mult(X, y / cval, i, j, dval, other)
    if inner is None:
        return None
    return ("mul", cnode, inner)


# ================= 总入口 =================

def structure_guided_solve(X, y, feats):
    _CACHE.clear()
    _POOL_CACHE.clear()
    feats = list(feats)
    cands = []
    pools = get_pools(X, feats)
    pr = peel(X, y, feats, pools)          # 快速路径（现有 115 题）
    if pr is not None:
        pnode = pr[1]
        pp = _ev_c(pnode, X)
        if pp is not None and _r2(y, pp) >= GATE and symbol_gate(pnode):
            return GATE, pnode            # 高质量真解，无需昂贵兜底
        cands.append(pnode)
    S = _state_surface(X, y)
    tn = _translation_path(X, y, feats, S)
    if tn is not None:
        cands.append(tn)
    mn = _mult_combo_path(X, y, feats, S)
    if mn is not None:
        cands.append(mn)
    ok = []
    for node in cands:
        pred = _ev_c(node, X)
        if pred is not None and _r2(y, pred) >= GATE:
            ok.append(node)
    if not ok:
        return pr
    gated = [n for n in ok if symbol_gate(n)]
    pick = gated if gated else ok
    pick.sort(key=_complexity)
    return GATE, pick[0]


def c2var(node):
    if isinstance(node, tuple):
        if node[0] == "c":
            return ("var", node[1])
        return tuple([node[0]] + [c2var(a) for a in node[1:]])
    return node


# ---------------- 符号门：全量 r2=1 后，零重算区分真解 vs 高精度展开 ----------------
def _add_terms(node):
    """add 链顶层项数（add 叶子数）。"""
    if isinstance(node, tuple) and node[0] == "add":
        return _add_terms(node[1]) + _add_terms(node[2])
    return 1


def _max_add_width(node):
    """表达式中所有 add 子树的最大顶层项数。"""
    best = _add_terms(node)
    if isinstance(node, tuple):
        for a in node[1:]:
            if isinstance(a, tuple):
                best = max(best, _max_add_width(a))
    return best


def _collect_consts(node, out):
    if isinstance(node, tuple):
        if node[0] == "const":
            out.append(float(node[1]))
        else:
            for a in node[1:]:
                _collect_consts(a, out)


def _coeff_simple(v, tol=1e-7):
    import sympy as sp
    if not np.isfinite(v):
        return False
    if abs(v) < 1e-12:
        return True
    r = sp.nsimplify(v, rational=True, tolerance=tol)
    if r.is_Rational and abs(r.q) <= 24:
        return True
    rp = sp.nsimplify(v, [sp.pi], tolerance=tol)
    if rp.has(sp.pi):
        # a·pi 形式：rp/pi 为有理数
        c1 = sp.nsimplify(rp / sp.pi, rational=True, tolerance=tol)
        if c1.is_Rational and abs(c1.q) <= 24:
            return True
        # a/pi 形式：rp*pi 为有理数
        c2 = sp.nsimplify(rp * sp.pi, rational=True, tolerance=tol)
        if c2.is_Rational and abs(c2.q) <= 24:
            return True
    return False


def symbol_gate(node):
    """全量解出后符号门：add 最大项数 ≤ MAX_TERMS，且所有系数为简单有理数/标准常数。"""
    if _max_add_width(node) > MAX_TERMS:
        return False
    cs = []
    _collect_consts(node, cs)
    for v in cs:
        if not _coeff_simple(v):
            return False
    return True
