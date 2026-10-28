# -*- coding: utf-8 -*-
"""统一 peel 库（G3 验证）：稳健 abs 探针 + 复合剥 + 剥后加性识别。"""
import sys
import numpy as np
from itertools import combinations

import probes as PB
import shr_v7_full as FULL
import shr_collapse as SC

GATE = 1 - 1e-7


def robust_abs_sweep(S, X, y, k, outlier_tol=0.15, min_curves=6):
    grid = S.grid_i(k)
    idx = np.random.default_rng(100 + k).choice(
        len(X), min(10, len(X)), replace=False)
    curves = [S.sweep(X[b], k, grid) for b in idx]
    ac = [np.abs(c) for c in curves]
    ref = len(curves[0]) // 2
    norms = [c / max(c[ref], 1e-9) for c in ac]
    med = np.median(np.array(norms), axis=0)
    devs = [np.max(np.abs(n - med)) / max(np.max(np.abs(med)), 1e-9)
            for n in norms]
    keep = [i for i, d in enumerate(devs) if d <= outlier_tol]
    if len(keep) < min_curves:
        return 0.0, None, 1.0
    err = 0.0
    for ai in range(len(keep)):
        for bi in range(ai + 1, len(keep)):
            a, b = keep[ai], keep[bi]
            err = max(err, np.max(np.abs(norms[a] - norms[b]))
                      / max(np.max(np.abs(norms[a])), 1e-9))
    ok = med > 1e-9
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
        c = 0.30 * (1 - min(err / 0.3, 1))
    return min(max(c, 0), 1), clean, err


def single_power_r2(X, yt, active, stt=None):
    actl = sorted(active)
    same = np.all(yt > 0) or np.all(yt < 0)
    if not same and stt is None:
        Xa = X[:, actl]
        stt = PB.State(Xa, yt, [('var', i) for i in range(Xa.shape[1])],
                       ["x%d" % i for i in actl])
    best = (-1, None, None)
    for k in active:
        x = X[:, k]
        if not same:
            Xa = X[:, actl]
            lk = actl.index(k)
            ca, pa, _ = robust_abs_sweep(stt.S, Xa, yt, lk)
            if ca > 0.8 and pa is not None and abs(pa - round(pa)) < 1e-9:
                p = int(round(pa))
                pred = np.power(np.clip(x, 1e-9, None), p)
                kk = np.sum(yt * pred) / np.sum(pred ** 2)
                rr = SC.r2(yt, kk * pred)
                if rr > best[0]:
                    best = (rr, k, p)
            continue
        for p in range(-6, 7):
            if p == 0 or (p < 0 and np.any(x <= 0)):
                continue
            pred = np.power(np.clip(x, 1e-9, None), p)
            if np.std(pred) < 1e-12:
                continue
            kk = np.sum(yt * pred) / np.sum(pred ** 2)
            rr = SC.r2(yt, kk * pred)
            if rr > best[0]:
                best = (rr, k, p)
    return best


def try_additive(self, X, yc, active, col_expr, peels):
    X1, cex1, deps1 = self._build_lean(X, col_expr)
    keep1 = [j for j in range(X1.shape[1]) if deps1[j] <= active]
    if keep1:
        s1, c1, r1 = self._additive(X1, yc, keep1, self.seed + 1, exact=False)
        if r1 >= GATE:
            e1, _ = self._assemble(s1, c1, cex1, col_expr, peels)
            return e1, peels
    X2, cex2, ce2, nm2, deps2 = self._b._build_all(X, yc, col_expr)
    keep2 = [j for j in range(X2.shape[1]) if deps2[j] <= active]
    if keep2:
        s2, c2, r2 = self._additive_fwd(X2, yc, keep2, self.seed + 2)
        if r2 >= GATE:
            e2, _ = self._assemble(s2, c2, cex2, col_expr, peels)
            return e2, peels
    return None


def build_const(self, yc, peels, col_expr):
    k = float(np.mean(yc))
    expr = ('const', k)
    for f in reversed(peels):
        expr = ('mul', f, expr)
    return expr, peels


def unified_peel(self, X, y, col_expr):
    active = frozenset(range(X.shape[1]))
    yc = y.copy()
    peels = []
    got = try_additive(self, X, yc, active, col_expr, peels)
    if got is not None:
        return got
    for _ in range(FULL.MAX_PEEL + 2):
        actl = sorted(active)
        Xa = X[:, actl]
        st = PB.State(Xa, yc, [('var', i) for i in range(len(actl))],
                      ["x%d" % i for i in actl], seed=self.seed)
        best_c, best_lk, best_p = 0.8, None, None
        for lk in range(len(actl)):
            c, p, _ = robust_abs_sweep(st.S, Xa, yc, lk)
            if c > best_c and p is not None and abs(p - round(p)) < 1e-9 \
                    and round(p) != 0:
                best_c, best_lk, best_p = c, lk, int(round(p))
        if best_lk is not None:
            orig_k = actl[best_lk]
            yt = yc / np.power(np.clip(X[:, orig_k], 1e-9, None), best_p)
            if not np.all(np.isfinite(yt)):
                break
            factor = col_expr[orig_k] if best_p == 1 else \
                ('pow', col_expr[orig_k], best_p)
            active = active - {orig_k}
            yc = yt
            peels = peels + [factor]
            self.log.append("剥 %s^%s" % (self.names0[orig_k], best_p))
            if np.std(yc) / max(np.mean(np.abs(yc)), 1e-9) < 1e-6:
                return build_const(self, yc, peels, col_expr)
            got = try_additive(self, X, yc, active, col_expr, peels)
            if got is not None:
                return got
            continue
        cands = []
        for a, b in combinations(actl, 2):
            comps = [
                (X[:, a] * X[:, b],
                 ('mul', col_expr[a], col_expr[b])),
                (X[:, a] ** 2 - X[:, b] ** 2,
                 ('sub', ('pow', col_expr[a], 2), ('pow', col_expr[b], 2))),
                (X[:, a] ** 2 + X[:, b] ** 2,
                 ('add', ('pow', col_expr[a], 2), ('pow', col_expr[b], 2))),
            ]
            for g, gexpr in comps:
                gz = np.clip(np.abs(g), 1e-9, None)
                for q in range(1, 6):
                    yt = yc / np.power(gz, q)
                    if np.all(np.isfinite(yt)) and \
                            (np.all(yt > 0) or np.all(yt < 0)):
                        rr, k, p = single_power_r2(X, yt, active)
                        if rr >= GATE:
                            cands.append((rr, yt, ('pow', gexpr, q)))
                    yt2 = yc * np.power(gz, q)
                    if np.all(np.isfinite(yt2)) and \
                            (np.all(yt2 > 0) or np.all(yt2 < 0)):
                        rr2, k2, p2 = single_power_r2(X, yt2, active)
                        if rr2 >= GATE:
                            cands.append((rr2, yt2, ('pow', gexpr, -q)))
        if not cands:
            break
        cands.sort(key=lambda t: -t[0])
        _, yc, factor = cands[0]
        peels = peels + [factor]
        self.log.append("复合剥 %s" % (factor,))
        if np.std(yc) / max(np.mean(np.abs(yc)), 1e-9) < 1e-6:
            return build_const(self, yc, peels, col_expr)
        got = try_additive(self, X, yc, active, col_expr, peels)
        if got is not None:
            return got
    return None


def patch():
    PB._abs_sweep_mult_conf = robust_abs_sweep
    FULL.SHRv7Full._build_const = build_const
    FULL.SHRv7Full._aggressive_path = unified_peel
