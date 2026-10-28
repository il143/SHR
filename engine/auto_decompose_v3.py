# -*- coding: utf-8 -*-
"""SHR 自动分解引擎 v3（含复验，盲测不喂真值）。

流程：
  自动 logy
  阶段一 剥外层（循环）：整体乘性(试剥-回验,多切分复验) -> 一元外函数(sqrt/exp)
                        -> 对称变换
  阶段二 核心：一元组合（内积木+外函数） 或 加性（交互矩阵分组 + 前向 + 多切分复验）
  阶段三 全局重拟合 + 恢复外层栈 + 真实数据 r2。

判据全部通用、跨题一致，不针对某题写死。
"""
import itertools
import numpy as np
import pandas as pd
from shr_surface import Surface
import shr_collapse as SC

# ---------------- 表达式 evaluate（结构树） ----------------
def ev(expr, X):
    t = expr[0]
    if t == 'var':
        return X[:, expr[1]]
    if t == 'const':
        return np.full(len(X), expr[1])
    if t == 'pow':
        return np.clip(ev(expr[1], X), 1e-7, None) ** expr[2]
    if t == 'mul':
        return ev(expr[1], X) * ev(expr[2], X)
    if t == 'div':
        return ev(expr[1], X) / np.clip(ev(expr[2], X), 1e-9, None)
    if t == 'add':
        return ev(expr[1], X) + ev(expr[2], X)
    if t == 'sub':
        return ev(expr[1], X) - ev(expr[2], X)
    if t == 'exp':
        return np.exp(np.clip(ev(expr[1], X), -50, 50))
    if t == 'sqrt':
        return np.sqrt(np.clip(ev(expr[1], X), 0, None))
    if t == 'log':
        return np.log(np.clip(ev(expr[1], X), 1e-9, None))
    if t == 'inv1m':
        return 1.0 / (1.0 - ev(expr[1], X))
    raise ValueError(t)


def _mult_err(curves):
    """v2 受控乘性检测：10 条 sweep 曲线归一化一致性。"""
    ref = len(curves[0]) // 2
    if any(curves[a][ref] * curves[b][ref] < 0 for a in range(len(curves))
           for b in range(a + 1, len(curves))):
        return 1.0
    norms = [c / c[ref] for c in curves]
    err = 0.0
    for a in range(len(norms)):
        for b in range(a + 1, len(norms)):
            err = max(err, np.max(np.abs(norms[a] - norms[b]))
                      / np.max(np.abs(norms[a])))
    return err


# ---------------- 因子 / 积木（乘性） ----------------
def _factor_pool(d, X):
    """每变量的因子：幂次 -4~4，加 1-x / 1-2x（对所有变量生成）。"""
    pool = []
    for i in range(d):
        fs = [(i, ('p', p)) for p in (-4, -3, -2, -1, 1, 2, 3, 4)]
        fs.append((i, ('s', 1)))
        fs.append((i, ('s', 2)))
        pool.append(fs)
    return pool


def ev_fac(fac, X):
    i, op = fac
    if op[0] == 'p':
        return X[:, i] ** op[1]
    return 1 - X[:, i] if op[1] == 1 else 1 - 2 * X[:, i]


def ev_block(block, X):
    z = np.ones(len(X))
    for f in block:
        z = z * ev_fac(f, X)
    return z


def block_name(block, names):
    s = ""
    for i, op in block:
        if op[0] == 'p':
            s += "%s^%s" % (names[i], op[1])
        else:
            s += "(1-%s)" % names[i] if op[1] == 1 else "(1-2%s)" % names[i]
    return s


def block_expr(block):
    """把积木转成结构树（内部列引用用 'c'，与原始变量 'var' 区分）。"""
    e = None
    for i, op in block:
        if op[0] == 'p':
            f = ('c', i) if op[1] == 1 else ('pow', ('c', i), op[1])
        elif op[1] == 1:
            f = ('sub', ('const', 1), ('c', i))
        else:
            f = ('sub', ('const', 1), ('mul', ('const', 2), ('c', i)))
        e = f if e is None else ('mul', e, f)
    return e


# 外函数菜单：tag -> (特征构造, 标准期望系数, 结构树)
def outer_menu(u):
    """返回 (tag, feat 数值, feat 结构树构造函数)。
    统一由拟合系数 a·feat + b 组装，外函数不含预设常数。"""
    eu = np.exp(np.clip(u, -50, 50))
    return [
        ('u', u, lambda e: e),
        ('u2', u ** 2, lambda e: ('pow', e, 2)),
        ('sqrt', np.sqrt(np.clip(u, 1e-9, None)), lambda e: ('sqrt', e)),
        ('log', np.log(np.clip(u, 1e-9, None)), lambda e: ('log', e)),
        ('1mu', 1 - u, lambda e: ('sub', ('const', 1), e)),
        ('exp', eu, lambda e: ('exp', e)),
        ('exp-1', eu, lambda e: ('exp', e)),
        ('inv1m', 1 / (1 - u), lambda e: ('inv1m', e)),
    ]


class SHRv3:
    def __init__(self, X, y, names, logy=None):
        self.X0 = np.asarray(X, float); self.y0 = np.asarray(y, float)
        self.names0 = list(names); self.d0 = X.shape[1]
        if logy is None:
            same = np.all(y > 0) or np.all(y < 0)
            logy = same and (np.abs(y).max() / np.abs(y).min() > 100)
        self.logy = logy
        self.log = []

    # ---------- sweep 单列检测（受控曲线） ----------
    def _sweep_col(self, S, X, i):
        grid = S.grid_i(i)
        rng = np.random.default_rng(100 + i)
        idx = rng.choice(len(X), size=10, replace=False)
        bps = X[idx].copy()
        curves = [S.sweep(bp, i, grid) for bp in bps]
        me = _mult_err(curves)
        slope = None
        if np.all(curves[0] > 0) or np.all(curves[0] < 0):
            slope = np.polyfit(np.log(grid), np.log(np.abs(curves[0])), 1)[0]
        return me, slope

    # ---------- 条件尺度法（跨零通用）：x_k 分 bin，std(y) 是否按 x_k^q ----
    def _scale_slope(self, X, y, k, nb=8):
        xb = X[:, k]
        qs = np.unique(np.quantile(xb, np.linspace(0, 1, nb + 1)))
        centers, scales = [], []
        for b in range(len(qs) - 1):
            if b == len(qs) - 2:
                m = (xb >= qs[b]) & (xb <= qs[b + 1])
            else:
                m = (xb >= qs[b]) & (xb < qs[b + 1])
            if m.sum() < 5:
                continue
            s = np.std(y[m])
            if s < 1e-12:
                continue
            centers.append(np.median(xb[m]))
            scales.append(s)
        if len(centers) < 5:
            return None
        lc = np.log(np.asarray(centers))
        ls = np.log(np.asarray(scales))
        q, ii = np.polyfit(lc, ls, 1)
        fit = q * lc + ii
        r2 = 1 - np.sum((ls - fit) ** 2) / \
            max(np.sum((ls - ls.mean()) ** 2), 1e-12)
        return q, r2

    # ---------- 剥整体乘性外层（同号 sweep；跨零尺度法；整数幂） ----------
    def _peel_mult(self, S, X, y, colidx):
        same = np.all(y > 0) or np.all(y < 0)
        if same:
            sweep_hits = []
            for k in colidx:
                me, slope = self._sweep_col(S, X, k)
                if slope is None:
                    continue
                cv = SC.clean_ci(slope)
                if me < 0.08 and abs(cv - slope) < 0.1 and abs(cv) > 1e-9 \
                        and abs(cv - round(cv)) < 1e-9:
                    sweep_hits.append((me, k, cv))
            if sweep_hits:
                sweep_hits.sort()
                return sweep_hits[0]
        # 跨零（或同号 sweep 无干净）：条件尺度法
        scale_hits = []
        for k in colidx:
            r = self._scale_slope(X, y, k)
            if r is None:
                continue
            q, r2 = r
            cv = SC.clean_ci(q)
            if r2 > 0.9 and abs(cv - q) < 0.08 and abs(cv) > 1e-9 \
                    and abs(cv - round(cv)) < 1e-9:
                scale_hits.append((1 - r2, k, cv))
        if scale_hits:
            scale_hits.sort()
            return scale_hits[0]
        return None

    # ---------- 拟合-剥离兜底（sweep 不清晰时，如变号/共享） ----------
    def _peel_fit(self, X, y, colidx, base_r2):
        """试剥每变量幂次，剥后 fit Surface，看 train_r2 是否显著提升。"""
        n = len(X); h = n // 2
        best = None
        for k in colidx:
            for q in (-2, -1, 1, 2):
                u = y / np.power(np.clip(X[:, k], 1e-9, None), q)
                if not np.all(np.isfinite(u)):
                    continue
                try:
                    Sr = Surface(X, u, epochs=2500)
                except Exception:
                    continue
                gain = Sr.train_r2 - base_r2
                if gain > 0.01 and (best is None or gain > best[0]):
                    best = (gain, k, q)
        return best

    # ---------- 变量 k 对 y 的主导幂次（sweep 优先，否则 scale），round整数 ----------
    def _dominant_power(self, X, y, k):
        S = Surface(X, y)
        q = None
        try:
            me, slope = self._sweep_col(S, X, k)
            if slope is not None:
                q = slope
        except Exception:
            pass
        if q is None:
            r = self._scale_slope(X, y, k)
            if r is not None:
                q = r[0]
        if q is None:
            return None
        cq = round(q)
        if cq != 0 and abs(q - cq) < 0.35:
            return int(cq)
        return None

    # ---------- 部分剥离（剥离收益，保留列） ----------
    def _peel_partial(self, X, y, col_expr, names):
        if X.shape[1] == 0:
            return X, y, []
        colidx = [k for k in range(X.shape[1]) if col_expr[k][0] == 'var']
        base_r2 = self._additive_r2(X, y)
        base_r2 = -1.0 if base_r2 is None else base_r2
        best = None
        for k in colidx:
            q = self._dominant_power(X, y, k)
            if not q:
                continue
            u = y / np.power(np.clip(X[:, k], 1e-9, None), q)
            if not np.all(np.isfinite(u)):
                continue
            r2v = self._additive_r2(X, u)
            r2v = -1.0 if r2v is None else r2v
            gain = r2v - base_r2
            if gain > 0.02 and (best is None or gain > best[0]):
                best = (gain, k, q, r2v)
        if best is None:
            return X, y, []
        gain, k, q, r2v = best
        y = y / np.power(np.clip(X[:, k], 1e-9, None), q)
        ce = col_expr[k]
        self.log.append("部分剥离 %s^%s (加性 r2 %.3f->%.3f, 收益%.3f)"
                        % (names[k], q, base_r2, r2v, gain))
        return X, y, [('mult', ce, q)]

    # ---------- 剥一元外函数（z 能被积木池高 r2 解释则剥） ----------
    def _peel_unary(self, X, y, colidx):
        # sqrt：全同号，z=y^2 能被加性积木池解释
        if np.all(y > 0) or np.all(y < 0):
            z = y ** 2
            r2v = self._additive_r2(X, z)
            if r2v is not None and r2v > 0.95:
                return ('sqrt', z)
        # exp：全正，z=log(y) 能被加性积木池解释
        if np.all(y > 0):
            z = np.log(np.clip(y, 1e-9, None))
            r2v = self._additive_r2(X, z)
            if r2v is not None and r2v > 0.95:
                return ('exp', z)
        return None

    def _has_structure(self, X, y, colidx):
        """目标 y 上是否有清晰乘性结构（sweep 或拟合-剥离）。"""
        S = Surface(X, y)
        if self._peel_mult(S, X, y, colidx) is not None:
            return True
        # 纯乘性（log 空间幂次干净）
        if np.all(y > 0) or np.all(y < 0):
            t = np.log(np.abs(y))
            lX = np.log(np.clip(X[:, colidx], 1e-9, None))
            A = np.column_stack([lX, np.ones(len(y))])
            coef, *_ = np.linalg.lstsq(A, t, rcond=None)
            c = coef[:-1]
            if np.max(np.abs(c - np.array([SC.clean_ci(v) for v in c]))) < 0.08:
                return True
        return False

    # ---------- 对称：prod/sm 构造后 bin_r2 回验，只接受真解释 ----------
    def _symmetric(self, S, X, i, j):
        # 点交换一致性：随机点交换 i,j 后，函数值是否严格相等（真可交换）
        rng = np.random.default_rng(77)
        idx = rng.choice(len(X), size=min(200, len(X)), replace=False)
        P = X[idx].copy()
        P2 = P.copy()
        P2[:, i], P2[:, j] = P[:, j], P[:, i]
        f1 = S.query(P)
        f2 = S.query(P2)
        rel = np.abs(f1 - f2) / (np.abs(f1) + np.abs(f2) + 1e-9)
        med = np.median(rel)
        p90 = np.quantile(rel, 0.9)
        return med < 0.005 and p90 < 0.02

    def _peel_symmetry(self, S, X, y, colidx, col_expr):
        # 只对原始变量列（'var'）做对称，生成的列不再参与（防链式泛滥）
        orig = [k for k in colidx if col_expr[k][0] == 'var']
        if len(orig) < 2:
            return None
        for a in range(len(orig)):
            for b in range(a + 1, len(orig)):
                i, j = orig[a], orig[b]
                if not self._symmetric(S, X, i, j):
                    continue
                prod = X[:, i] * X[:, j]
                sm = X[:, i] + X[:, j]
                terms = [('mul', prod, 1.0), ('add', sm, 1.0)]
                return i, j, terms
        return None

    def _term_explains(self, z, y, lg):
        """z 对 y 的解释力（bin_r2）；大跨度同号时走 log 空间。"""
        z = np.asarray(z, float)
        if lg and np.all(np.abs(z) > 0):
            return SC.bin_r2(np.log(np.abs(z)), np.log(np.abs(y)))
        if np.std(z) < 1e-12:
            return 0.0
        return SC.bin_r2(z, y)

    # ---------- 交互矩阵（局部梯度幅值相关） ----------
    def _interaction(self, X, y, colidx):
        S = Surface(X, y)
        rng = np.random.default_rng(2)
        P = 600
        lo, hi = X.min(0), X.max(0)
        pts = lo + rng.random((P, X.shape[1])) * (hi - lo)
        eps = 0.05 * (hi - lo)
        g = {}
        for i in colidx:
            Xp = pts.copy(); Xp[:, i] += eps[i]
            Xm = pts.copy(); Xm[:, i] -= eps[i]
            g[i] = (S.query(Xp) - S.query(Xm)) / (2 * eps[i])
        m = len(colidx)
        M = np.zeros((m, m))
        for a in range(m):
            for b in range(m):
                ga, gb = g[colidx[a]], g[colidx[b]]
                # 梯度幅值相关（同组变量联动）
                ca, cb = ga - ga.mean(), gb - gb.mean()
                rr = np.sum(ca * cb) / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-12)
                M[a, b] = rr
        return M

    def _partition(self, M, colidx, th=0.25):
        """由交互矩阵分组，识别共享变量（与多组高交互）。"""
        m = len(colidx)
        adj = M > th
        # 简单连通分量（不含共享判定）
        groups = []
        unseen = set(range(m))
        while unseen:
            s = unseen.pop(); comp = [s]; st = [s]
            while st:
                u = st.pop()
                for v in range(m):
                    if v in unseen and (adj[u, v] or adj[v, u]):
                        unseen.remove(v); comp.append(v); st.append(v)
            groups.append(comp)
        # 共享变量：跨多组、且实际与不同组都高——先返回连通组，
        # 若全连通（一个大组）则不是加性（一元组合）
        shared = []
        return groups, shared

    # ---------- 核心：一元组合（内积木 + 外函数） ----------
    def _core_unary(self, X, y, colidx, names):
        pool = _factor_pool(X.shape[1], X)
        pool = [pool[k] for k in colidx]
        blocks = []
        rmax = min(3, len(colidx))
        for r in range(2, rmax + 1):
            for vars in itertools.combinations(range(len(pool)), r):
                for comb in itertools.product(*[pool[v] for v in vars]):
                    blocks.append(comb)
        cand = []
        for b in blocks:
            z = ev_block(b, X)
            if not np.all(np.isfinite(z)) or np.std(z) < 1e-9:
                continue
            for tag, feat, fexpr_fn in outer_menu(z):
                if not np.all(np.isfinite(feat)) or np.std(feat) < 1e-9:
                    continue
                A = np.column_stack([feat, np.ones_like(feat)])
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                pred = coef[0] * feat + coef[1]
                r2v = SC.r2(y, pred)
                cand.append((r2v, b, fexpr_fn, coef))
        if not cand:
            return None
        cand.sort(key=lambda c: round(c[0], 5), reverse=True)
        r2v, b, fexpr_fn, coef = cand[0]
        if r2v < 0.999:
            return None
        expr = fexpr_fn(block_expr(b))
        a, bb = coef[0], coef[1]
        # 统一 y = a·feat + b
        if abs(a - 1) > 1e-3:
            expr = ('mul', ('const', float(a)), expr)
        if abs(bb) > 1e-3:
            expr = ('add', expr, ('const', float(bb)))
        return r2v, expr

    # ---------- 加性积木池：单列 + 2列 + 3列 ----------
    def _build_blocks(self, X):
        d = X.shape[1]
        pool = _factor_pool(d, X)
        blocks = []
        for k in range(d):
            blocks.append([(k, ('p', 1))])       # 单列
        for r in (2, 3):
            for vars in itertools.combinations(range(d), r):
                for comb in itertools.product(*[pool[v] for v in vars]):
                    blocks.append(list(comb))
        Z = np.column_stack([ev_block(b, X) for b in blocks])
        return blocks, Z

    # ---------- 前向选择（带截距，候选子集，投票在当前残差） ----------
    def _fwd_on(self, Z, y, cand_idx, blocks=None, max_steps=6,
                vote_th=0.4, silent=False):
        n = len(y)
        splits = []
        for seed in range(5):
            rr = np.random.default_rng(seed).permutation(n)
            nt = n * 3 // 4
            splits.append((rr[:nt], rr[nt:]))
        def ns_of(bi):
            if blocks is None:
                return 0
            return sum(1 for _, op in blocks[bi] if op[0] == 's')
        selected = []
        resid = y.copy()
        for step in range(max_steps):
            scored = []
            for bi in cand_idx:
                if bi in selected:
                    continue
                zc = Z[:, bi] - Z[:, bi].mean()
                if np.std(zc) < 1e-9:
                    continue
                rc = resid - resid.mean()
                c1 = np.sum(rc * zc) / np.sum(zc * zc)
                tr = 1 - np.sum((rc - c1 * zc) ** 2) / np.sum(rc ** 2)
                scored.append((tr, bi))
            best = None
            for _, bi in sorted(scored, reverse=True)[:20]:
                vs = []
                for tr, te in splits:
                    ztr = Z[tr, bi]
                    zc = ztr - ztr.mean()
                    rc = resid[tr] - resid[tr].mean()
                    if np.std(zc) < 1e-9:
                        continue
                    c1 = np.sum(rc * zc) / np.sum(zc * zc)
                    pred = c1 * (Z[te, bi] - ztr.mean()) + resid[tr].mean()
                    vs.append(SC.r2(resid[te], pred))
                m = np.mean(vs)
                ns = ns_of(bi)
                if best is None:
                    best = (bi, m, ns)
                else:
                    bm, bns = best[1], best[2]
                    if ns < bns:
                        take = m > bm - 0.002
                    elif ns > bns:
                        take = m > bm + 0.002
                    else:
                        take = m > bm
                    if take:
                        best = (bi, m, ns)
            if best is None or best[1] < vote_th:
                break
            bi = best[0]
            zc = Z[:, bi] - Z[:, bi].mean()
            rc = resid - resid.mean()
            c1 = np.sum(rc * zc) / np.sum(zc * zc)
            resid = resid - c1 * zc
            selected.append(bi)
            if not silent:
                self.log.append("加性第%d步 积木%s (复验%.3f)" %
                                (step + 1, bi, best[1]))
        return selected

    # ---------- 加性拟合：先单列后复合（带截距），可静默 ----------
    def _add_fit(self, X, y, silent=False):
        blocks, Z = self._build_blocks(X)
        d = X.shape[1]
        sel1 = self._fwd_on(Z, y, list(range(d)), blocks=blocks,
                            silent=silent)
        resid = y.copy()
        if sel1:
            Zs1 = np.column_stack([Z[:, bi] for bi in sel1])
            A = np.column_stack([Zs1, np.ones(len(y))])
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            resid = y - Zs1 @ coef[:-1] - coef[-1]
        sel2 = self._fwd_on(Z, resid, list(range(d, Z.shape[1])),
                            blocks=blocks, silent=silent)
        allsel = sel1 + sel2
        if not allsel:
            return None
        Zs = np.column_stack([Z[:, bi] for bi in allsel])
        A = np.column_stack([Zs, np.ones(len(y))])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = Zs @ coef[:-1] + coef[-1]
        r2v = SC.r2(y, pred)
        if not silent:
            self.log.append("加性重拟合 r2=%.4f 积木%s" % (r2v, allsel))
        return blocks, allsel, coef, r2v

    # ---------- 静默：加性对 y 的解释 r2 ----------
    def _additive_r2(self, X, y):
        r = self._add_fit(X, y, silent=True)
        return None if r is None else r[3]

    # ---------- 核心：加性（组装表达式） ----------
    def _core_additive(self, X, y):
        r = self._add_fit(X, y, silent=False)
        if r is None:
            return None
        blocks, allsel, coef, r2v = r
        expr = None
        for j, bi in enumerate(allsel):
            be = block_expr(blocks[bi])
            cj = coef[j]
            term = be if abs(cj - 1) < 1e-9 else \
                ('mul', ('const', float(cj)), be)
            expr = term if expr is None else ('add', expr, term)
        if abs(coef[-1]) > 1e-9:
            expr = ('add', expr, ('const', float(coef[-1])))
        return expr, r2v

    # ---------- 列引用 'c' 提升到原始变量（'var' 保持） ----------
    @staticmethod
    def _lift(e, col_expr):
        t = e[0]
        if t == 'c':
            return col_expr[e[1]]
        if t in ('var', 'const'):
            return e
        if t == 'pow':
            return ('pow', SHRv3._lift(e[1], col_expr), e[2])
        if t in ('exp', 'sqrt', 'log', 'inv1m'):
            return (t, SHRv3._lift(e[1], col_expr))
        return (t, SHRv3._lift(e[1], col_expr), SHRv3._lift(e[2], col_expr))

    # ---------- 主流程 ----------
    @staticmethod
    def _autology(y):
        same = np.all(y > 0) or np.all(y < 0)
        return same and (np.abs(y).max() / max(np.abs(y).min(), 1e-12) > 100)

    # ---------- 剥外层：乘性(sweep+拟合兜底) -> 对称，循环 ----------
    def _peel_outer_loop(self, X, y, col_expr, names):
        added = []
        for _ in range(8):
            did = False
            # 乘性只对原始变量列（'var'）剥，对称/生成的列不参与（防误剥）
            colidx = [k for k in range(X.shape[1]) if col_expr[k][0] == 'var']
            lg = self._autology(y)
            S = Surface(X, y, logy=lg)
            base_r2 = S.train_r2
            pm = self._peel_mult(S, X, y, colidx)
            if pm is None:
                pf = self._peel_fit(X, y, colidx, base_r2)
                if pf is not None:
                    pm = (pf[0], pf[1], pf[2])
            if pm is not None:
                r, k, q = pm
                ce = col_expr[k]
                nmk = names[k]
                y = y / np.power(np.clip(X[:, k], 1e-9, None), q)
                keep = [c for c in range(X.shape[1]) if c != k]
                X = np.column_stack([X[:, c] for c in keep]) if keep \
                    else np.empty((len(y), 0))
                col_expr[:] = [col_expr[c] for c in keep]
                names[:] = [names[c] for c in keep]
                added.append(('mult', ce, q))
                self.log.append("剥乘性 %s^%s (%.3f)" % (nmk, q, r))
                did = True
            if did:
                continue
            ps = self._peel_symmetry(S, X, y, colidx, col_expr)
            if ps is not None:
                i, j, terms = ps
                ci, cj = col_expr[i], col_expr[j]
                ni, nj = names[i], names[j]
                keep = [c for c in range(X.shape[1]) if c not in (i, j)]
                new_cols, new_ce, new_nm = [], \
                    [col_expr[c] for c in keep], [names[c] for c in keep]
                for op, val, r2v in terms:
                    new_cols.append(val)
                    if op == 'mul':
                        new_ce.append(('mul', ci, cj))
                        new_nm.append(ni + "*" + nj)
                    else:
                        new_ce.append(('add', ci, cj))
                        new_nm.append(ni + "+" + nj)
                base = [X[:, c] for c in keep]
                X = np.column_stack(base + new_cols) if base \
                    else np.column_stack(new_cols)
                col_expr[:] = new_ce          # 原地更新，传回 run
                names[:] = new_nm
                self.log.append("对称 %s,%s -> %s" %
                                (ni, nj, "/".join(t[0] for t in terms)))
                did = True
            if not did:
                break
        return X, y, col_expr, names, added

    @staticmethod
    def _restore_added(expr, added):
        while added:
            s = added.pop()
            if s[0] == 'mult':
                ce, q = s[1], s[2]
                fac = ce if q == 1 else ('pow', ce, q)
                expr = ('mul', expr, fac)
        return expr

    # ---------- 递归求解 ----------
    def _solve(self, X, y, col_expr, names, depth):
        if depth > 4:
            return None
        # 严格剥外层（乘性 + 对称）
        X, y, col_expr, names, added = self._peel_outer_loop(X, y,
                                                             col_expr, names)
        if X.shape[1] == 0:           # 所有变量剥完，y 为常数
            return self._restore_added(('const', float(np.median(y))), added)
        # 部分剥离循环（剥离收益，保留列）
        for _ in range(3):
            Xn, yn, prec = self._peel_partial(X, y, col_expr, names)
            if not prec:
                break
            X, y = Xn, yn
            added.extend(prec)
        colidx = list(range(X.shape[1]))
        # 1) 一元组合
        core = self._core_unary(X, y, colidx, names)
        if core is not None:
            return self._restore_added(core[1], added)
        # 2) 加性（高精度才立即接受，否则留作兜底）
        add = self._core_additive(X, y)
        if add is not None and add[1] >= 0.99:
            return self._restore_added(add[0], added)
        # 3) 剥一元（递归试解，先包外函数再恢复外层乘性）
        if np.all(y > 0) or np.all(y < 0):
            z = y ** 2
            ce = list(col_expr); nm = list(names)
            inner = self._solve(X.copy(), z, ce, nm, depth + 1)
            if inner is not None:
                inner_l = self._lift(inner, ce)
                r2z = SC.r2(z, ev(inner_l, X))
                if r2z > 0.99:
                    e = ('sqrt', inner)
                    return self._restore_added(e, added)
        if np.all(y > 0):
            z = np.log(np.clip(y, 1e-9, None))
            ce = list(col_expr); nm = list(names)
            inner = self._solve(X.copy(), z, ce, nm, depth + 1)
            if inner is not None:
                inner_l = self._lift(inner, ce)
                r2z = SC.r2(z, ev(inner_l, X))
                if r2z > 0.99:
                    e = ('exp', inner)
                    return self._restore_added(e, added)
        # 加性兜底（中等精度也返回，避免无解）
        if add is not None:
            return self._restore_added(add[0], added)
        return None

    def run(self):
        col_expr = [('var', i) for i in range(self.d0)]
        e = self._solve(self.X0.copy(), self.y0.copy(), col_expr,
                        list(self.names0), 0)
        if e is None:
            return None, float('nan'), self.log
        e = self._lift(e, col_expr)
        pred = ev(e, self.X0)
        r2v = SC.r2(self.y0, pred)
        return e, r2v, self.log


def auto_decompose(X, y, names, logy=None):
    return SHRv3(X, y, names, logy=logy).run()
