# -*- coding: utf-8 -*-
"""SHR criterion-driven recursive solver.

Data is split into train/val, then the loop runs:
  A) Power fit (first, deterministic): target = k * prod(X^c). Outer exp is
     peeled (nested=0,1,2); coefficients c are solved by least squares in log
     space and accepted only when they are clean integers/simple fractions
     and the recovered val r2 passes (otherwise rejected as approximation).
  B) Residual relevance: each candidate column is scored against the
     train/val residuals; consistently relevant columns are kept (capped).
  C) Feature generation: unary sq/exp/log/sqrt/sin/cos and pairwise
     sub/mul/div; only consistently relevant new columns are added.
  D) Forward selection: least squares on train, validated on val.
  E) Unary outer: a column z with stable bin_r2 to y matches a basic unary.
  F) Outcome: solved / capability_limit (consecutive stalls) / budget end.

The search stays bounded: powers solve analytically (O(d^2)); the active
set is capped; candidates are deduplicated; rounds are limited.
"""
import numpy as np
from itertools import combinations

# ---------- 口径旋钮 ----------
REL_TH = 0.05
UNI_TH = 0.99
MAX_ACTIVE = 8
NEW_M = 8
BLOCK_KEEP = 40
CLEAN_TOL = 0.06     # 系数接近干净值的容差
GAIN_STALL = 1e-3
STALL_N = 3
MAX_ROUNDS = 6
MAX_NEST = 2
SOLVE_R2 = 0.999
MAX_TERMS = 12
NB = 12
_SAFE = 1e6


# ================= 基础判据 =================
def bin_r2(z, y, nb=NB):
    z = np.asarray(z, float); y = np.asarray(y, float)
    ok = np.isfinite(z) & np.isfinite(y)
    z, yy = z[ok], y[ok]
    n = len(yy)
    if n < nb * 2 or np.var(yy) <= 1e-14:
        return 0.0
    ys = yy[np.argsort(z)]
    edges = np.linspace(0, n, nb + 1).astype(int)
    sse = 0.0
    for k in range(nb):
        seg = ys[edges[k]:edges[k + 1]]
        if len(seg):
            sse += ((seg - seg.mean()) ** 2).sum()
    sst = ((yy - yy.mean()) ** 2).sum()
    return float(max(0.0, 1 - sse / max(sst, 1e-12)))


def r2(y, p):
    p = np.nan_to_num(p, nan=0.0, posinf=_SAFE, neginf=-_SAFE)
    sst = ((y - y.mean()) ** 2).sum()
    if sst <= 1e-14:
        return 1.0 if np.allclose(y, p) else 0.0
    return float(1 - ((y - p) ** 2).sum() / sst)


def _safe_col(c):
    c = np.asarray(c, float)
    if np.all(np.isfinite(c)) and np.max(np.abs(c)) < _SAFE:
        return c
    return None


def clean_ci(x):
    """最近的干净值（分母≤4 的分数，分子 ±8）。"""
    best, bestd = 0.0, 1e9
    for den in range(1, 5):
        for num in range(-8, 9):
            v = num / den
            d = abs(x - v)
            if d < bestd:
                bestd, best = d, v
    return best


# ================= A) Power fit =================
def power_nested(y, X, tr, va):
    # 目标符号：全正 sign=1；全负 sign=-1（对 |y| 做）；有正有负 -> 不适用
    if np.all(y > 0):
        sign = 1.0
    elif np.all(y < 0):
        sign = -1.0
    else:
        return None
    yabs = np.abs(y)
    for nest in range(MAX_NEST + 1):
        t = yabs.copy()
        ok = True
        for _ in range(nest):
            if np.min(t) <= 1e-9:
                ok = False; break
            t = np.log(t)
        if not ok or np.min(t) <= 1e-9 or np.min(X) <= 1e-9:
            continue
        lX, lt = np.log(X), np.log(t)
        A = np.column_stack([lX, np.ones(len(y))])
        coef, *_ = np.linalg.lstsq(A[tr], lt[tr], rcond=None)
        c, b = coef[:-1], coef[-1]
        cc = np.array([clean_ci(v) for v in c])
        if np.max(np.abs(c - cc)) > CLEAN_TOL:
            continue
        t_pred = np.exp(b) * np.prod(X ** cc, axis=1)
        y_pred = t_pred
        for _ in range(nest):
            y_pred = np.exp(np.clip(y_pred, -50, 50))
        y_pred = sign * y_pred
        vr = r2(y[va], y_pred[va])
        if vr >= SOLVE_R2:
            return dict(val_r2=vr, nest=nest, c=np.round(cc, 3),
                        b=round(float(b), 4), sign=sign,
                        expr=("" if sign > 0 else "-") + _power_expr(cc, b, nest))
    return None


def _power_expr(cc, b, nest):
    parts = []
    for i, ci in enumerate(cc):
        if abs(ci) > 1e-9:
            parts.append("x%d^%s" % (i + 1,
                         str(int(ci)) if abs(ci - int(ci)) < 1e-9 else str(ci)))
    s = ("%.3g*" % np.exp(b)) + "*".join(parts)
    for _ in range(nest):
        s = "exp(%s)" % s
    return s


# ================= 几何 =================
MAX_UNI_DEPTH = 3   # 一元参数深度上限（允许 exp(mul)、exp(qV/kT)）
MAX_BIN_DEPTH = 2   # 两两结果深度上限（参数 max≤1；允许 qV/kT，禁 sub(sub(...))）


def gen_layer(active):
    out = {}

    def put(key, c, depth):
        c = _safe_col(c)
        if c is not None:
            out[key] = (c, depth)

    for nm, col, key, dep in active:
        if dep < MAX_UNI_DEPTH:
            put("sq(%s)" % key, col ** 2, dep + 1)
            put("exp(%s)" % key, np.exp(np.clip(col, -50, 50)), dep + 1)
            put("sin(%s)" % key, np.sin(col), dep + 1)
            put("cos(%s)" % key, np.cos(col), dep + 1)
            if np.min(col) > 1e-9:
                put("log(%s)" % key, np.log(col), dep + 1)
                put("sqrt(%s)" % key, np.sqrt(col), dep + 1)
    for (na, ca, ka, da), (nb2, cb, kb, db) in combinations(active, 2):
        if max(da, db) >= MAX_BIN_DEPTH:
            continue
        nd = max(da, db) + 1
        ka2, kb2 = sorted([ka, kb])
        put("add(%s,%s)" % (ka2, kb2), ca + cb, nd)
        put("sub(%s,%s)" % (ka, kb), ca - cb, nd)
        put("sub(%s,%s)" % (kb, ka), cb - ca, nd)
        put("mul(%s,%s)" % (ka2, kb2), ca * cb, nd)
        if np.min(np.abs(cb)) > 1e-9:
            put("div(%s,%s)" % (ka, kb), ca / cb, nd)
        if np.min(np.abs(ca)) > 1e-9:
            put("div(%s,%s)" % (kb, ka), cb / ca, nd)
    return out


# ================= D) 前向选择 =================
def fwd_select_tv(cols, y, tr, va, max_terms=MAX_TERMS):
    names = list(cols)
    X = np.column_stack([cols[n][0] for n in names])
    Xtr, Xva = X[tr], X[va]
    ytr, yva = y[tr], y[va]
    mu = Xtr.mean(0)
    Xc = Xtr - mu
    norms = np.linalg.norm(Xc, axis=0, keepdims=True)
    Xn = Xc / np.clip(norms, 1e-12, None)
    yc = ytr - ytr.mean()
    sel, resid = [], yc.copy()
    for _ in range(max_terms):
        corr = Xn.T @ resid
        if sel:
            corr[sel] = 0.0
        j = int(np.argmax(np.abs(corr)))
        sel.append(j)
        beta, *_ = np.linalg.lstsq(Xc[:, sel], yc, rcond=None)
        resid = yc - Xc[:, sel] @ beta
    beta, *_ = np.linalg.lstsq(Xc[:, sel], yc, rcond=None)
    pred_tr = Xc[:, sel] @ beta + ytr.mean()
    pred_va = (Xva - mu)[:, sel] @ beta + ytr.mean()
    return [names[j] for j in sel], r2(ytr, pred_tr), r2(yva, pred_va), pred_tr, pred_va


def _ls_r2(ftr, ytr, fva, yva):
    A = np.column_stack([ftr, np.ones_like(ftr)])
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return r2(yva, coef[0] * fva + coef[1])


# ================= Recursive fit =================
class Collapse:
    def __init__(self, X, y, seed=0, max_rounds=MAX_ROUNDS):
        self.X = np.asarray(X, float); self.y = np.asarray(y, float)
        self.max_rounds = max_rounds
        n, d = self.X.shape
        perm = np.random.default_rng(seed).permutation(n)
        self.tr, self.va = perm[:n // 2], perm[n // 2:]
        self.cols = {}; self.seen = set(); self.orig = set()
        for i in range(d):
            k = "x%d" % (i + 1)
            self.cols[k] = (self.X[:, i], k, 0); self.seen.add(k); self.orig.add(k)
        self.d = d; self.zseq = 0; self.trace = []
        self.adopted = set(); self.uni_blocks = set()
        self.best_va = -1.0; self.stall = 0
        m0 = self.y[self.tr].mean()
        self.pred_tr = np.full(len(self.tr), m0)
        self.pred_va = np.full(len(self.va), m0)

    def _new_name(self):
        self.zseq += 1
        return "z%d" % self.zseq

    def _stable(self, col, rt, rv):
        return min(bin_r2(col[self.tr], rt), bin_r2(col[self.va], rv))

    def _active(self, rt, rv):
        scored = sorted(((self._stable(self.cols[n][0], rt, rv), n)
                        for n in self.cols), reverse=True)
        names = [n for s, n in scored if s > REL_TH][:MAX_ACTIVE]
        return [(n, self.cols[n][0], self.cols[n][1], self.cols[n][2]) for n in names]

    def _uni_collapse(self):
        new_uni = []
        for name in list(self.cols):
            col, key, dep = self.cols[name]
            rel_y = min(bin_r2(col[self.tr], self.y[self.tr]),
                        bin_r2(col[self.va], self.y[self.va]))
            if rel_y < UNI_TH:
                continue
            u = gen_layer([(name, col, key, dep)])
            for uk, (ucol, ud) in u.items():
                vr = _ls_r2(ucol[self.tr], self.y[self.tr],
                            ucol[self.va], self.y[self.va])
                if vr >= SOLVE_R2:
                    return dict(val_r2=vr, expr=uk)
                if uk not in self.seen:
                    new_uni.append((uk, ucol, ud))
        for uk, ucol, ud in new_uni:
            if uk not in self.seen:
                self.cols[self._new_name()] = (ucol, uk, ud)
                self.seen.add(uk); self.uni_blocks.add(uk)
        return None

    def _prune(self, rt, rv, sel):
        for n in sel:
            if n not in self.orig:
                self.adopted.add(n)
        protected = self.orig | self.adopted
        protected |= set(n for n in self.cols if self.cols[n][1] in self.uni_blocks)
        room = BLOCK_KEEP - len(protected)
        mid = sorted(((self._stable(self.cols[n][0], rt, rv), n)
                     for n in self.cols if n not in protected), reverse=True)
        keep = set(n for s, n in mid[:max(0, room)])
        for n in list(self.cols):
            if n not in protected and n not in keep:
                del self.cols[n]

    def run(self):
        # A) Power fit (first, independent of round)
        hit = power_nested(self.y, self.X, self.tr, self.va)
        if hit is not None:
            self.trace.append(dict(round=1, active=0, gen=0, added=0,
                                   cols=len(self.cols), val_r2=hit["val_r2"],
                                   stall=0, expr=hit["expr"]))
            return "solved", [hit["expr"]], hit["val_r2"]
        for rnd in range(1, self.max_rounds + 1):
            rt = self.y[self.tr] - self.pred_tr
            rv = self.y[self.va] - self.pred_va
            active = self._active(rt, rv)
            gtr = gen_layer([(n, c[self.tr], k, dp) for n, c, k, dp in active])
            gva = gen_layer([(n, c[self.va], k, dp) for n, c, k, dp in active])
            gfu = gen_layer(active)
            keys = (set(gtr) & set(gva) & set(gfu)) - self.seen
            scored = sorted(((min(bin_r2(gtr[k][0], rt), bin_r2(gva[k][0], rv)), k)
                            for k in keys), reverse=True)
            n_added = 0
            for s, k in scored[:NEW_M]:
                if s > REL_TH:
                    col, dp = gfu[k]
                    self.cols[self._new_name()] = (col, k, dp)
                    self.seen.add(k); n_added += 1
            # E) 一元外层
            uh = self._uni_collapse()
            if uh is not None:
                return "solved", [uh["expr"]], uh["val_r2"]
            # D) 前向选择
            sel, rtr, rva, ptr, pva = fwd_select_tv(self.cols, self.y, self.tr, self.va)
            self.pred_tr, self.pred_va = ptr, pva
            gain = rva - self.best_va
            if gain > GAIN_STALL:
                self.best_va, self.stall = rva, 0
            else:
                self.stall += 1
            self._prune(rt, rv, sel)
            self.trace.append(dict(round=rnd, active=len(active), gen=len(gfu),
                                   added=n_added, cols=len(self.cols),
                                   val_r2=rva, stall=self.stall))
            if rva >= SOLVE_R2:
                return "solved", sel, rva
            if self.stall >= STALL_N:
                return "capability_limit", sel, rva
        sel, rtr, rva, *_ = fwd_select_tv(self.cols, self.y, self.tr, self.va)
        return "budget_exhausted", sel, rva


def collapse(X, y, seed=0, **kw):
    c = Collapse(X, y, seed=seed, **kw)
    state, sel, score = c.run()
    return state, score, sel, c.trace
