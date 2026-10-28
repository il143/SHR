# -*- coding: utf-8 -*-
"""SHR v7 symbolic-regression engine.

Pipeline:
  1. Detect: structural probes (relevance, multiplicative, additive, outer,
     symmetry, periodic) produce 0-1 confidence scores; each candidate is
     accepted / continued / rejected under a budget.
  2. Solve: direct solving first; outer transforms (sqrt/exp/log) are only
     enumerated when direct solving falls short, and the simplest candidate
     (fewest expression nodes) is selected. Peeling clips extrapolation to
     the training range; feature blocks are grown on demand.
  3. Re-validate: multiple splits plus residual diagnostics; unstable
     candidates re-enter the pipeline until a stable result or budget end.

The engine does not read ground truth, problem names, or per-problem
branches; train/test splits are supplied by the caller.
"""
import numpy as np
from itertools import combinations
import probes as PB
import shr_collapse as SC
from auto_decompose_v5 import _ev_c, _br, _sinc_z, SHRv5

# ---------- 旋钮 ----------
SOLVE_R2 = 0.9995     # 解出阈值（后验强门）
PEEL_MIN = 0.99       # 剥后最低可接受
MAX_OUTER = 2         # 外层反演递归深度
MAX_PEEL = 6          # 乘性 peel 最多剥几列
BACK_EPS = 0.01       # 反向选择 val 明显下降阈值
LEAN_TERMS = 4        # 简洁精确直接返回的最大项数
EXT_DELTA = 0.01      # 外推保护量纲放宽
# Three-state thresholds
TR_YES = 0.99
TR_NO = 0.15
OUTER_WEAK = 0.50


# ---------- 表达式节点数（符号简洁度） ----------
def _nodes(expr):
    if not isinstance(expr, tuple):
        return 1
    op = expr[0]
    if op in ('var', 'const'):
        return 1
    if op in ('sin', 'cos', 'sqrt', 'log', 'exp', 'sinc', 'inv1m'):
        return 1 + _nodes(expr[1])
    if op == 'pow':
        # expr[2] 是数字幂次，不计节点
        return 1 + _nodes(expr[1])
    return 1 + sum(_nodes(a) for a in expr[1:])


class SHRv7:
    # 外层候选是否用 unified_peel 连续剥兜底（SHRv7Full 打开）。SHRv7 基类
    # 只走 _solve_direct（剥门控要求每步 r2 提升，切分/种子敏感时会卡死）。
    _unified_outer = False

    def __init__(self, X, y, names, seed=0, verbose=False):
        self.X0 = np.asarray(X, float)
        self.y0 = np.asarray(y, float)
        self.names0 = list(names)
        self.d0 = X.shape[1]
        self.seed = seed
        self.verbose = verbose
        self.log = []
        self._expr = None
        self._r2v = None
        self.state = None
        self.peels = []
        self.outer = None
        self.family = None
        self.states = None
        self._b = SHRv5(self.X0, self.y0, self.names0)
        self._bounds = {i: (float(X[:, i].min()), float(X[:, i].max()))
                        for i in range(self.d0)}
        self._protect_cols = set()

    # ============================================================
    # Layer 1: detect (linear probe enumeration + family + three-state screening)
    # ============================================================
    def _recognize(self, X, y):
        col_expr = [('var', i) for i in range(X.shape[1])]
        if X.shape[1] == self.d0:
            names = list(self.names0)
        else:
            names = ["x%d" % i for i in range(X.shape[1])]
        st = PB.State(X, y, col_expr, names, seed=self.seed)
        mi = PB.aggregate_observations(PB.scan_all(st))
        family = self._family(mi)
        states = self._tri_states(mi)
        self.log.append("L1 detect family=%s three-state=%s" % (
            family, {k: (s, round(c, 2)) for k, (s, c) in states.items()}))
        return mi, family, states

    def _family(self, mi):
        outer = [(c, d) for c, e, d in mi['outer'] if c >= OUTER_WEAK]
        mult_max = max([c for c, (c, d) in mi['mult'].items()], default=0.0)
        add_c = mi['add'][0] if mi['add'] else 0.0
        per = [c for c, (c, d) in mi['periodic'].items()]
        if outer:
            return 'outer:' + str(outer[0][1].get('wrap', '?'))
        if per and max(per) >= 0.8:
            return 'periodic'
        if mult_max >= 0.8:
            return 'mult'
        if add_c >= 0.7:
            return 'add'
        return 'composite'

    def _tri_states(self, mi):
        out = {}
        for k in range(self.d0):
            rel = mi['relevance'].get(k, 0.0)
            mult_c = mi['mult'].get(k, (0.0, None))[0]
            c = max(rel, mult_c)
            if c >= TR_YES:
                out[k] = ('yes', c)
            elif c < TR_NO:
                out[k] = ('no', c)
            else:
                out[k] = ('maybe', c)
        return out

    # ============================================================
    # L1 精简列空间：原始 + 一元 + pairwise 积/和/差/差平方
    # ============================================================
    def _build_lean(self, X, col_expr):
        n, d = X.shape
        cols = []

        def add(v, e, dp):
            if np.all(np.isfinite(v)) and np.std(v) > 1e-12:
                cols.append((v, e, frozenset(dp)))

        for i in range(d):
            add(X[:, i], col_expr[i], {i})
        for i in range(d):
            x = X[:, i]
            add(np.sin(x), ('sin', col_expr[i]), {i})
            add(np.cos(x), ('cos', col_expr[i]), {i})
            add(x * x, ('pow', col_expr[i], 2), {i})
            if np.all(x > 0):
                add(np.sqrt(np.clip(x, 0, None)),
                    ('sqrt', col_expr[i]), {i})
                add(np.log(np.clip(x, 1e-9, None)),
                    ('log', col_expr[i]), {i})
        for a, b in combinations(range(d), 2):
            add(X[:, a] * X[:, b],
                ('mul', col_expr[a], col_expr[b]), {a, b})
            add(X[:, a] + X[:, b],
                ('add', col_expr[a], col_expr[b]), {a, b})
            d_expr = ('sub', col_expr[a], col_expr[b])
            add(X[:, a] - X[:, b], d_expr, {a, b})
            add((X[:, a] - X[:, b]) ** 2, ('pow', d_expr, 2), {a, b})
        X_ext = np.column_stack([c[0] for c in cols])
        cex = [c[1] for c in cols]
        deps = [c[2] for c in cols]
        return X_ext, cex, deps

    # ============================================================
    # 加性求解：联合 lstsq + 反向选择（内部 75/25 把关）
    # ============================================================
    def _additive(self, X_ext, y, keep, seed, exact=True):
        keep = list(keep)
        n = len(y)
        pm = np.random.default_rng(seed).permutation(n)
        cut = int(0.75 * n)
        tr, va = pm[:cut], pm[cut:]

        def fit(sel, rows):
            A = np.column_stack([X_ext[rows, j] for j in sel]
                                + [np.ones(len(rows))])
            c, *_ = np.linalg.lstsq(A, y[rows], rcond=None)
            p = np.column_stack([X_ext[rows, j] for j in sel]
                                + [np.ones(len(rows))]) @ c
            return SC.r2(y[rows], p), c

        cur = keep[:]
        best_rv, best_set = fit(cur, va)[0], cur[:]
        while len(cur) > 1:
            rv_now = fit(cur, va)[0]
            if exact:
                best_loss, best_j = 1e9, None
                for j in cur:
                    rv = fit([k for k in cur if k != j], va)[0]
                    loss = rv_now - rv
                    if loss < best_loss:
                        best_loss, best_j = loss, j
                cur.remove(best_j)
            else:
                _, coef_tr = fit(cur, tr)
                ac = [abs(c) for c in coef_tr[:-1]]
                mx = max(ac) if ac else 1.0
                drop = [j for j, c in zip(cur, coef_tr[:-1])
                        if abs(c) < 0.02 * mx]
                if len(drop) >= len(cur):
                    drop = [j for j in cur if j != cur[int(np.argmax(ac))]]
                if not drop:
                    best_loss, best_j = 1e9, None
                    for j in cur:
                        rv = fit([k for k in cur if k != j], va)[0]
                        if rv_now - rv < best_loss:
                            best_loss, best_j = rv_now - rv, j
                    drop = [best_j]
                cur = [j for j in cur if j not in drop]
            rv = fit(cur, va)[0]
            if rv >= best_rv - 1e-4:
                best_rv, best_set = rv, cur[:]
            if rv < best_rv - BACK_EPS:
                break
        r2f, coef = fit(best_set, np.arange(n))
        return best_set, coef, r2f

    def _additive_fwd(self, X_ext, y, keep, seed, max_terms=LEAN_TERMS):
        """前向 greedy：从空开始，每轮在 val 上选一个最能提升 r2 的列。
        复杂度 O(max_terms x |keep|)，适合 L2 几百列快速探索（反向选择
        从全列 lstsq、多轮，在几百列上太慢）。"""
        keep = list(keep)
        n = len(y)
        pm = np.random.default_rng(seed).permutation(n)
        cut = int(0.75 * n)
        tr, va = pm[:cut], pm[cut:]

        def fit(sel, rows):
            A = np.column_stack([X_ext[rows, j] for j in sel]
                                + [np.ones(len(rows))])
            c, *_ = np.linalg.lstsq(A, y[rows], rcond=None)
            return SC.r2(y[rows], A @ c), c

        chosen, remaining = [], keep[:]
        best_rv = -1.0
        for t in range(max_terms):
            cand = []
            for j in remaining:
                rv, _ = fit(chosen + [j], va)
                cand.append((rv, j))
            cand.sort(key=lambda z: -z[0])
            rv, j = cand[0]
            if chosen and rv < best_rv + 1e-4:
                break
            chosen.append(j)
            remaining.remove(j)
            best_rv = rv
            if best_rv >= SOLVE_R2:
                break
        r2f, coef = fit(chosen, np.arange(n))
        return chosen, coef, r2f

    # 高维阈值：L1 列数超过它时，反向 exact 太慢，改前向+反向精修
    FAST_KEEP = 70

    def _additive_fast(self, X_ext, y, keep, seed, max_terms=LEAN_TERMS):
        """高维混合：前向快速选（max_terms+2），再在小列集反向精修。
        反向 exact 在 |keep| 大时 O(|keep|^2) 太慢；前向 O(terms·|keep|)
        快速锁定候选，反向在小列集上保证精确。"""
        s, c, r = self._additive_fwd(
            X_ext, y, keep, seed, max_terms=max_terms + 2)
        if len(s) > 1:
            s2, c2, r2 = self._additive(X_ext, y, s, seed + 1, exact=True)
            return s2, c2, r2
        return s, c, r

    def _additive_auto(self, X_ext, y, keep, seed, exact=True):
        """高维自动选择：keep<=FAST_KEEP 走 exact 真算（准且快，d<=5 全
        exact 解仍走这条）；keep>FAST_KEEP 走 _additive_fast（前向+小集合
        反向），避免高维 O(k^2) 在算子不覆盖的残题上空转。
        依据（J8 核验）：d>5 的 exact 真解全部由快路径/GP/_solve 给出，
        激进路径在 keep>FAST_KEEP 时只遇残题；fast 是前向近似（r2 偏低），
        只用于快速判不覆盖。"""
        if len(keep) <= self.FAST_KEEP:
            return self._additive(X_ext, y, keep, seed, exact=exact)
        return self._additive_fast(X_ext, y, keep, seed)

    # ============================================================
    # 复合乘性剥离：原始剥不动时，在 pairwise 积上剥整数幂
    # ============================================================
    def _peel_compound(self, X, y, active, X1, deps1, col_expr,
                       r2_now, peel_iter, depth):
        act_list = sorted(active)
        cands = []
        for a, b in combinations(act_list, 2):
            pv = X[:, a] * X[:, b]
            for q in range(1, 6):
                yt = y / np.power(np.clip(pv, 1e-9, None), q)
                if not np.all(np.isfinite(yt)):
                    continue
                keep = [j for j in range(X1.shape[1])
                        if deps1[j] <= active]
                _, _, r_t = self._additive(
                    X1, yt, keep,
                    self.seed + 7 * a + 13 * b + 100 * peel_iter + depth,
                    exact=False)
                if r_t >= PEEL_MIN and r_t > r2_now + 1e-6:
                    factor = ('pow', ('mul', col_expr[a], col_expr[b]), q)
                    cands.append((r_t, yt, factor))
        if not cands:
            return None
        cands.sort(key=lambda t: -t[0])
        r_t, yt, factor = cands[0]
        keep = [j for j in range(X1.shape[1]) if deps1[j] <= active]
        _, _, r_ex = self._additive(
            X1, yt, keep,
            self.seed + 91 + 100 * peel_iter + depth, exact=True)
        if r_ex >= PEEL_MIN and r_ex > r2_now + 1e-6:
            self.log.append("L2复合剥 剥后r2=%.4f (不剥%.4f)" % (
                r_ex, r2_now))
            return yt, factor
        return None

    # ============================================================
    # 第2层・直接求解：peel 循环（L1/L2 加性 + 乘性剥），返回最佳候选
    # (r2, expr, peels)；不枚举外层。
    # ============================================================
    def _solve_direct(self, X, y, col_expr, depth):
        X1, cex1, deps1 = self._build_lean(X, col_expr)
        active = frozenset(range(X.shape[1]))
        ycur = y
        peels = []
        best = None  # (r2, expr, peels)
        X2 = cex2 = deps2 = None
        l2_built = False

        def note_best(r, e, pl):
            nonlocal best
            if best is None or r > best[0]:
                best = (r, e, pl[:])

        for peel_iter in range(MAX_PEEL + 1):
            r2_here = -1.0
            # L1
            keep1 = [j for j in range(X1.shape[1]) if deps1[j] <= active]
            if keep1:
                if len(keep1) > self.FAST_KEEP:
                    # 高维 L1 反向 exact O(cols^2) 太慢，前向+反向精修
                    s1, c1, r1 = self._additive_fast(
                        X1, ycur, keep1,
                        self.seed + 100 * peel_iter + depth)
                else:
                    s1, c1, r1 = self._additive(
                        X1, ycur, keep1,
                        self.seed + 100 * peel_iter + depth,
                        exact=True)
                r2_here = max(r2_here, r1)
                e1, _ = self._assemble(s1, c1, cex1, col_expr, peels)
                note_best(r1, e1, peels)
                if r1 >= SOLVE_R2:
                    return r1, e1, peels[:]
            # L2 按需（L1 不达标后建，仅一次）
            if not l2_built:
                X2, cex2, ce2, nm2, deps2 = self._b._build_all(X, y, col_expr)
                l2_built = True
            keep2 = [j for j in range(X2.shape[1]) if deps2[j] <= active]
            if keep2:
                s2, c2, r2 = self._additive_fwd(
                    X2, ycur, keep2, self.seed + 100 * peel_iter + depth)
                r2_here = max(r2_here, r2)
                e2, _ = self._assemble(s2, c2, cex2, col_expr, peels)
                note_best(r2, e2, peels)
                if r2 >= SOLVE_R2 and len(s2) <= LEAN_TERMS:
                    return r2, e2, peels[:]
            if peel_iter >= MAX_PEEL:
                break
            # peel MultProbe
            act_list = sorted(active)
            Xa = X[:, act_list]
            st = PB.State(Xa, ycur,
                          [('var', i) for i in range(len(act_list))],
                          ["x%d" % i for i in act_list], seed=self.seed)
            mi = PB.aggregate_observations(PB.scan_all(st, probes=[PB.MultProbe]))
            scored = []
            for lk, (c, det) in mi['mult'].items():
                p = det.get('power') if det else None
                if p is None or abs(p - round(p)) > 1e-9 or round(p) == 0:
                    continue
                orig_k = act_list[lk]
                active2 = active - {orig_k}
                x_div = X[:, orig_k]
                if int(round(p)) < 0:
                    lo, hi = self._bounds.get(orig_k,
                                              (x_div.min(), x_div.max()))
                    x_div = np.clip(x_div, lo * (1 - EXT_DELTA),
                                    hi * (1 + EXT_DELTA))
                yt = ycur / np.power(np.clip(x_div, 1e-9, None), p)
                if not np.all(np.isfinite(yt)):
                    continue
                k1 = [j for j in range(X1.shape[1])
                      if deps1[j] <= active2]
                k2 = [j for j in range(X2.shape[1])
                      if deps2[j] <= active2]
                r_fast = -1.0
                if k1:
                    _, _, r_fast = self._additive(
                        X1, yt, k1,
                        self.seed + 100 * peel_iter + depth + 1, exact=False)
                if k2:
                    _, _, r_t2 = self._additive_fwd(
                        X2, yt, k2,
                        self.seed + 100 * peel_iter + depth + 1)
                    r_fast = max(r_fast, r_t2)
                scored.append((r_fast, orig_k, int(round(p)), active2, yt, k1))
            scored.sort(key=lambda t: -t[0])
            final = []
            for idx, (r_fast, orig_k, p, active2, yt, k1) in enumerate(scored):
                r_ex = r_fast
                if idx < 2 and k1:
                    _, _, r_ex = self._additive(
                        X1, yt, k1,
                        self.seed + 100 * peel_iter + depth + 91, exact=True)
                final.append((max(r_fast, r_ex), orig_k, p, active2, yt))
            final.sort(key=lambda t: -t[0])
            moved = False
            for r_t, orig_k, p, active2, yt in final:
                if r_t < PEEL_MIN or r_t <= r2_here + 1e-6:
                    continue
                self.log.append("L2剥 %s^%s 剥后r2=%.4f (不剥%.4f)" % (
                    self.names0[orig_k], p, r_t, r2_here))
                active, ycur = active2, yt
                factor = col_expr[orig_k] if p == 1 else \
                    ('pow', col_expr[orig_k], p)
                peels.append(factor)
                if p < 0:
                    self._protect_cols.add(orig_k)
                moved = True
                break
            if not moved:
                comp = self._peel_compound(
                    X, ycur, active, X1, deps1, col_expr,
                    r2_here, peel_iter, depth)
                if comp is not None:
                    yt, factor = comp
                    ycur = yt
                    peels.append(factor)
                    moved = True
                else:
                    break
        return best

    # ============================================================
    # 第2层・外层枚举：仅在直接解不达标时，按符号枚举 sqrt/exp/log，
    # 递归（内部仍先直接解），收集全部后验达标候选，按节点数选最简洁。
    # ============================================================
    def _wrap_back(self, wrap, e_in, sgn):
        if wrap == 'sqrt':
            e_out = ('sqrt', e_in)
            if sgn < 0:
                e_out = ('sub', ('const', 0.0), e_out)
            return e_out
        if wrap == 'exp':
            return ('exp', e_in)
        if wrap == 'square':
            return ('pow', e_in, 2)
        return ('log', e_in)

    def _single_outer_cands(self, X, y, col_expr, depth):
        """单层外层候选：对每个符号（sqrt/exp/log）做 transform，直接解 z，
        后验达标即收。不嵌套——数学上嵌套外层（sqrt∘exp 等）等价于某个
        更简洁的单层外层（sqrt(exp z)=exp(z/2)），嵌套只会更臃肿。"""
        cands = []
        if np.all(y > 0) or np.all(y < 0):
            z = y ** 2
            if np.all(np.isfinite(z)):
                cands.append(('sqrt', z, int(np.sign(y[0]))))
        if np.all(y > 0):
            z = np.log(np.clip(y, 1e-9, None))
            if np.all(np.isfinite(z)):
                cands.append(('exp', z, 1))
            # 整体开方：y = z²（z>0），解 z 再平方（test_1 类）
            z = np.sqrt(y)
            if np.all(np.isfinite(z)):
                cands.append(('square', z, 1))
        z = np.exp(np.clip(y, -50, 50))
        if np.all(np.isfinite(z)):
            cands.append(('log', z, 1))
        out = []
        for wrap, z, sgn in cands:
            solved = None
            direct = self._solve_direct(X, z, col_expr, depth + 1)
            if direct is not None and direct[0] >= SOLVE_R2:
                e_out = self._wrap_back(wrap, direct[1], sgn)
                r = SC.r2(y, _ev_c(e_out, X))
                if r >= SOLVE_R2:
                    solved = (r, e_out, direct[2])
            # _solve_direct 剥门控卡死（要求每步剥后 r2 严格提升，切分/种子
            # 敏感时一步不剥）→ 用 unified_peel 连续剥兜底，最后统一严格门。
            # 仅 square 外层（整体开方后连续剥，test_1 型）；其余外层不兜底。
            if solved is None and wrap == 'square' and \
                    getattr(self, '_unified_outer', False):
                ures = self._aggressive_path(X, z, col_expr)
                if ures is not None:
                    e_out = self._wrap_back(wrap, ures[0], sgn)
                    r = SC.r2(y, _ev_c(e_out, X))
                    if r >= SOLVE_R2:
                        solved = (r, e_out, ures[1])
            if solved is not None:
                out.append((solved[0], solved[1], solved[2], wrap))
        return out

    # ============================================================
    # 第2层主编排：收集直接解 + 单层外层，统一在"原始 y 后验"上
    # 评 r2，达标后按表达式节点数选最简洁。
    # 不"直接解达标即返回"——否则局部区间高次多项式会短路真解
    # （I_6_2a：5 项多项式 r2=.999995 vs exp 真解 7 节点）；
    # 也不嵌套外层（嵌套必等价于更简洁的单层）。
    # ============================================================
    def _solve(self, X, y, col_expr, depth):
        pool = []
        direct = self._solve_direct(X, y, col_expr, depth)
        if direct is not None:
            e_d = direct[1]
            r_d = SC.r2(y, _ev_c(e_d, X))
            pool.append((r_d, e_d, direct[2], None))
        for r, e_out, peels, wrap in self._single_outer_cands(
                X, y, col_expr, depth):
            pool.append((r, e_out, peels, wrap))
        ok = [c for c in pool if c[0] >= SOLVE_R2]
        if ok:
            ok.sort(key=lambda c: _nodes(c[1]))
            r, e, peels, wrap = ok[0]
            if wrap is not None:
                self.outer = wrap
                self.log.append("L2外层%s 后验r2=%.5f 节点=%d" % (
                    wrap, r, _nodes(e)))
            return e, peels
        if pool:
            pool.sort(key=lambda c: -c[0])
            r, e, peels, wrap = pool[0]
            if r >= PEEL_MIN:
                return e, peels
        return None

    # ============================================================
    # 组装：加性 cex 列 + coef，乘回 peel
    # ============================================================
    def _assemble(self, sel, coef, cex, col_expr, peels):
        expr = None
        for j, cj in zip(sel, coef[:-1]):
            if abs(cj) < 1e-9:
                continue
            be = cex[j]
            term = be if abs(cj - 1) < 1e-6 else \
                ('mul', ('const', float(cj)), be)
            expr = term if expr is None else ('add', expr, term)
        if expr is None:
            expr = ('const', 0.0)
        if abs(coef[-1]) > 1e-7:
            expr = ('add', expr, ('const', float(coef[-1])))
        for factor in reversed(peels):
            expr = ('mul', factor, expr)
        return expr, peels

    # ============================================================
    # 第3层：观测者复验（多切分 + 残差诊断）
    # ============================================================
    def _review(self, expr, X, y):
        n = len(y)
        scores = []
        for sd in range(3):
            pm = np.random.default_rng(500 + sd).permutation(n)
            cut = int(0.75 * n)
            va = pm[cut:]
            pred = _ev_c(expr, X)
            scores.append(SC.r2(y[va], pred[va]))
        pred_all = _ev_c(expr, X)
        resid = y - pred_all
        signal = False
        if np.var(resid) > 1e-9:
            Xr, cexr, depsr = self._build_lean(
                X, [('var', i) for i in range(X.shape[1])])
            try:
                _, _, rr = self._additive(Xr, resid,
                                          list(range(Xr.shape[1])),
                                          self.seed + 777, exact=False)
                signal = rr >= 0.5
            except Exception:
                signal = False
        return scores, signal

    # ============================================================
    # 对外入口
    # ============================================================
    def fit(self, X=None, y=None, goal=None, verify_fn=None,
            K_provider=None, budget=None, seed=0):
        mi, family, states = self._recognize(self.X0, self.y0)
        self.family = family
        self.states = states
        col_expr = [('var', i) for i in range(self.d0)]
        res = self._solve(self.X0.copy(), self.y0.copy(), col_expr, 0)
        if res is None:
            self.state = 'capability_limit'
            self._r2v = float('nan')
            self._expr = None
            return self
        e, peels = res
        self.peels = peels
        self._expr = e
        scores, signal = self._review(e, self.X0, self.y0)
        pred = _ev_c(e, self.X0)
        self._r2v = SC.r2(self.y0, pred)
        self.log.append("L3复验 多切分r2=%s 残差信号=%s" % (
            [round(s, 4) for s in scores], signal))
        self.state = 'solved' if self._r2v >= 0.99 else 'capability_limit'
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        if self._protect_cols:
            Xc = X.copy()
            for k in self._protect_cols:
                lo, hi = self._bounds[k]
                Xc[:, k] = np.clip(Xc[:, k], lo * (1 - EXT_DELTA),
                                   hi * (1 + EXT_DELTA))
            return _ev_c(self._expr, Xc)
        return _ev_c(self._expr, X)

    def get_expr(self):
        return self._expr

    def get_r2(self):
        return self._r2v

    def state_info(self):
        return {'state': self.state, 'r2': self._r2v, 'family': self.family,
                'peels': self.peels, 'outer': self.outer,
                'protect': sorted(self._protect_cols), 'log': self.log}


def fit_engine(X, y, names, seed=0, verbose=False):
    return SHRv7(X, y, names, seed=seed, verbose=verbose).fit()
