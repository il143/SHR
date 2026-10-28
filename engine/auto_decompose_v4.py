# -*- coding: utf-8 -*-
"""SHR v4：探针候选 + 试剥-验证（有边界穷举）+ 剥层重扫 + 外层递归。

表达式基准（修复列引用错位）：
- 每层 _solve_r 同时维护两套列表达式：
  - col_expr：X0 表达式（'var' 引用原始 X0 列），最终 run 使用；
  - ce_local：相对"本层输入 X"的表达式（'var' 引用本层 X），用于验证。
  剥层时两者做完全相同的更新。
- core 解出的表达式用 'c'（引用当前剥层 X），由两套映射分别 lift。
- 外层递归验证直接用内层返回的数值预测 pred（在共同的行上），不重新 lift，
  从而避开基准错位。

探针只"锁小空间+排序"，确认靠试剥（测量）——不取消穷举，只把 M 锁成 K。
"""
import numpy as np
from shr_surface import Surface
import probes as PB
from auto_decompose_v3 import SHRv3, ev_block, block_expr, _factor_pool
import shr_collapse as SC

GAIN_MULT = 0.15
GAIN_SYM = 0.88
VERIFY_EPOCHS = 1200

# 边际引导加性分解（先定维后定项）
MARG_ETA = 0.10    # 边际 eta^2 阈值：真实成分≥0.29，无关≤0.022
MARG_VOTE = 0.20   # 已确认成分后的前向投票阈值（单变量对 d 变量和 r2≈1/d）


# ---------- 表达式辅助 ----------
def _lift(e, mapping):
    """'c'/'var' i -> mapping[i]，递归。"""
    t = e[0]
    if t in ('c', 'var'):
        return mapping[e[1]]
    if t == 'const':
        return e
    if t == 'pow':
        return ('pow', _lift(e[1], mapping), e[2])
    if t in ('exp', 'sqrt', 'log', 'inv1m'):
        return (t, _lift(e[1], mapping))
    return (t, _lift(e[1], mapping), _lift(e[2], mapping))


def _ev_c(e, X):
    """在 X 上求值；'c'/'var' 均取 X 列。"""
    t = e[0]
    if t in ('c', 'var'):
        return X[:, e[1]]
    if t == 'const':
        return np.full(len(X), e[1])
    if t == 'pow':
        return np.clip(_ev_c(e[1], X), 1e-7, None) ** e[2]
    if t == 'mul':
        return _ev_c(e[1], X) * _ev_c(e[2], X)
    if t == 'div':
        return _ev_c(e[1], X) / np.clip(_ev_c(e[2], X), 1e-9, None)
    if t == 'add':
        return _ev_c(e[1], X) + _ev_c(e[2], X)
    if t == 'sub':
        return _ev_c(e[1], X) - _ev_c(e[2], X)
    if t == 'exp':
        return np.exp(np.clip(_ev_c(e[1], X), -50, 50))
    if t == 'sqrt':
        return np.sqrt(np.clip(_ev_c(e[1], X), 0, None))
    if t == 'log':
        return np.log(np.clip(_ev_c(e[1], X), 1e-9, None))
    if t == 'inv1m':
        return 1.0 / (1.0 - _ev_c(e[1], X))
    raise ValueError(t)


class SHRv4(SHRv3):

    def _verify_r2(self, Xp, t, logy=None):
        if Xp.shape[1] == 0:
            return None
        S = Surface(Xp, t, logy=logy, epochs=VERIFY_EPOCHS)
        return S.train_r2

    # ============ 前置剥层（探针候选，平行维护 col_expr / ce_local） ============
    def _peel_local(self, X, y, col_expr, ce_local, nm_local):
        """col_expr（X0）与 ce_local（本层 X）平行更新。mult 优先、直接剥
        （探针 sweep 硬判据 c>=0.9，复刻 v3）；mult 剥不动才 sym 兜底。
        返回 (X,y,col_expr,ce_local,nm_local,added,added_local)。"""
        added, added_local = [], []
        for layer in range(10):
            st = PB.State(X, y, ce_local, nm_local)
            mi = PB.aggregate_observations(PB.scan_all(st, probes=[PB.MultProbe,
                                                       PB.SymProbe]))
            # 1) mult 高置信度直接剥（sweep 已是多点测量）
            mc = []
            for k, (c, detail) in mi['mult'].items():
                p = detail.get('power') if detail else None
                if p is not None and c >= 0.9 and round(p) != 0 \
                        and abs(p - round(p)) < 1e-9:
                    mc.append((c, k, int(round(p))))
            if mc:
                mc.sort(reverse=True)
                c, k, p = mc[0]
                nmk = nm_local[k]
                ce_rm, cel_rm = col_expr[k], ce_local[k]
                keep = [cc for cc in range(X.shape[1]) if cc != k]
                yn = y / np.power(np.clip(X[:, k], 1e-9, None), p)
                if np.all(np.isfinite(yn)):
                    Xn = X[:, keep] if keep else np.empty((len(y), 0))
                    col_expr[:] = [col_expr[cc] for cc in keep]
                    ce_local[:] = [ce_local[cc] for cc in keep]
                    nm_local[:] = [nm_local[cc] for cc in keep]
                    added.append(('mult', ce_rm, p))
                    added_local.append(('mult', cel_rm, p))
                    X, y = Xn, yn
                    self.log.append("v4剥层 mult %s^%s (c=%.2f)" % (nmk, p, c))
                    continue
            # 2) sym 兜底（先判断 y 是否单列加性可表达：平方和走此路，
            #    不做 sym 替换、保留原始列，交给后续 core_additive）
            # 标定：平方和单列 r2≈0.972；纯 prod≈0.93、exp prod≈0.87
            if self._single_col_r2(X, y) >= 0.96:
                break
            sc = [(c, evd) for c, evd, detail in mi['sym'] if evd and c >= 0.9]
            sc.sort(reverse=True)
            did = False
            if sc:
                r = self._try_sym_local(sc[0][1], X, y, col_expr,
                                        ce_local, nm_local, st)
                if r is not None:
                    X, y = r['X'], r['y']
                    col_expr[:], ce_local[:], nm_local[:] = \
                        r['ce'], r['cel'], r['nm']
                    self.log.append("v4剥层 %s (r2=%.3f)" % (r['desc'], r['gain']))
                    did = True
            if not did:
                break
        return X, y, col_expr, ce_local, nm_local, added, added_local

    def _try_sym_local(self, evd, X, y, col_expr, ce_local, nm_local, st):
        i, j = evd['i'], evd['j']
        if self._symmetric(st.S, X, i, j) < 0.9:
            return None
        prod = X[:, i] * X[:, j]
        sm = X[:, i] + X[:, j]
        # 平方和对称（单列 x² 加性可表达）已在 peel_local 被拦在此前，
        # 这里只处理真正的两列聚合：prod / sm（2 列正好，不膨胀）
        cands = [
            (prod, ('mul', col_expr[i], col_expr[j]),
                   ('mul', ce_local[i], ce_local[j]),
                   nm_local[i] + '*' + nm_local[j]),
            (sm, ('add', col_expr[i], col_expr[j]),
                 ('add', ce_local[i], ce_local[j]),
                 nm_local[i] + '+' + nm_local[j]),
        ]
        keep = [c for c in range(X.shape[1]) if c not in (i, j)]
        base = [X[:, c] for c in keep]
        Xn = np.column_stack(base + [c[0] for c in cands])
        ce = [col_expr[c] for c in keep] + [c[1] for c in cands]
        cel = [ce_local[c] for c in keep] + [c[2] for c in cands]
        nm = [nm_local[c] for c in keep] + [c[3] for c in cands]
        return {'X': Xn, 'y': y, 'ce': ce, 'cel': cel, 'nm': nm,
                'added': None, 'added_local': None,
                'gain': 1.0,
                'desc': "sym %s,%s -> %s" % (
                    nm_local[i], nm_local[j],
                    ','.join(c[3] for c in cands))}

    # ============ 部分剥离（不删列，平行维护两套） ============
    def _dominant_power(self, X, y, k, S=None):
        """v4：支持外部传入 Surface（部分剥离多变量复用）。"""
        if S is None:
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

    def _single_col_r2(self, X, y):
        """单列变换（含幂次）的加性 r2，纯 numpy（不训练 Surface）。
        用于判断 y 是否能被各列自身变换后相加表达（如平方和）。"""
        d = X.shape[1]
        pool = _factor_pool(d, X)
        blocks, Zc = [], []
        for k in range(d):
            for fac in pool[k]:
                b = [(k, fac)]
                z = ev_block(b, X)
                if np.all(np.isfinite(z)) and np.std(z) > 1e-9:
                    blocks.append(b)
                    Zc.append(z)
        if not blocks:
            return -1.0
        Zm = np.column_stack(Zc)
        sel = self._fwd_on(Zm, y, list(range(len(blocks))),
                           blocks=blocks, vote_th=0.30, silent=True)
        if not sel:
            return -1.0
        Zs = np.column_stack([Zm[:, i] for i in sel])
        A = np.column_stack([Zs, np.ones(len(y))])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = Zs @ coef[:-1] + coef[-1]
        return SC.r2(y, pred)

    def _additive_r2(self, X, y):
        """部分剥离筛选用：快速通道（无 Surface）的加性 r2。"""
        r = self._add_fit(X, y, silent=True, fast=True)
        return None if r is None else r[3]

    def _peel_partial_local(self, X, y, col_expr, ce_local, nm_local):
        if X.shape[1] == 0:
            return X, y, [], []
        orig = [k for k in range(X.shape[1]) if ce_local[k][0] == 'var']
        base_r2 = self._additive_r2(X, y)
        base_r2 = -1.0 if base_r2 is None else base_r2
        best = None
        if orig:
            S = Surface(X, y)
        for k in orig:
            q = self._dominant_power(X, y, k, S=S)
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
            return X, y, [], []
        gain, k, q, r2v = best
        y = y / np.power(np.clip(X[:, k], 1e-9, None), q)
        self.log.append("部分剥离 %s^%s (加性 %.3f->%.3f, +%.3f)"
                        % (nm_local[k], q, base_r2, r2v, gain))
        return X, y, [('mult', col_expr[k], q)], [('mult', ce_local[k], q)]

    # ============ 核心求解 ============
    def _solve(self, X, y, col_expr, names, depth):
        e, _, _ = self._solve_r(X, y, col_expr, names, depth)
        return e

    # ============ 边际引导加性分解（先定维、后定项） ============
    def _marginal_profile(self, x, y, nb=12):
        """bin 条件期望边际：返回 (eta^2, centers, mu)。"""
        bins = np.linspace(x.min(), x.max(), nb + 1)
        idx = np.clip(np.digitize(x, bins) - 1, 0, nb - 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
        mu = np.zeros(nb)
        for b in range(nb):
            m = idx == b
            if m.sum():
                mu[b] = y[m].mean()
        return SC.r2(y, mu[idx]), centers, mu

    def _fit_marginal_forms(self, centers, mu):
        """候选 fac 对边际的拟合 r2，返回排序 [(r2, fac)]。"""
        res = []
        for p in (-4, -3, -2, -1, 1, 2, 3, 4):
            zc = np.clip(centers, 1e-9, None) ** p
            A = np.column_stack([zc, np.ones(len(mu))])
            c, *_ = np.linalg.lstsq(A, mu, rcond=None)
            res.append((SC.r2(mu, zc * c[0] + c[1]), ('p', p)))
        for n in (1, 2):
            zc = 1 - n * centers
            if np.std(zc) < 1e-9:
                continue
            A = np.column_stack([zc, np.ones(len(mu))])
            c, *_ = np.linalg.lstsq(A, mu, rcond=None)
            res.append((SC.r2(mu, zc * c[0] + c[1]), ('s', n)))
        res.sort(reverse=True)
        return res

    def _add_fit_marginal(self, X, y, silent=False):
        """边际引导：每轮对变量算边际 eta^2 定维、对边际拟合定项；
        投影残差迭代（backfitting）。返回 (blocks, selected, coef, r2)。"""
        d = X.shape[1]
        blocks, Zcols = [], []

        def add_block(b):
            z = ev_block(b, X)
            if np.all(np.isfinite(z)) and np.std(z) > 1e-9:
                blocks.append(b)
                Zcols.append(z)
                return len(blocks) - 1
            return None

        selected = []

        def resid():
            if not selected:
                return y.copy()
            Zs = np.column_stack([Zcols[i] for i in selected])
            A = np.column_stack([Zs, np.ones(len(y))])
            c, *_ = np.linalg.lstsq(A, y, rcond=None)
            return y - Zs @ c[:-1] - c[-1]

        def r2cur():
            return SC.r2(y, y - resid()) if selected else -1.0

        for _ in range(6):
            if r2cur() >= 0.9999:
                break
            r = resid()
            cand = []
            for k in range(d):
                eta, centers, mu = self._marginal_profile(X[:, k], r)
                if eta < MARG_ETA:
                    continue
                forms = self._fit_marginal_forms(centers, mu)
                r2f, fac = forms[0]
                b = [(k, fac)]
                if any(blocks[i] == b for i in range(len(blocks))):
                    continue
                bi = add_block(b)
                if bi is not None:
                    cand.append(bi)
            if not cand:
                break
            sel = self._fwd_on(np.column_stack(Zcols), r, cand,
                               blocks=blocks, vote_th=MARG_VOTE, silent=silent)
            if not sel:
                break
            selected.extend(sel)

        if not selected:
            return None
        Zs = np.column_stack([Zcols[i] for i in selected])
        A = np.column_stack([Zs, np.ones(len(y))])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = Zs @ coef[:-1] + coef[-1]
        r2v = SC.r2(y, pred)
        if not silent:
            self.log.append("边际引导 积木=%d 选用=%d r2=%.4f"
                            % (len(blocks), len(selected), r2v))
        self.last_nblocks = len(blocks)
        return blocks, selected, coef, r2v

    def _core_additive(self, X, y):
        """优先边际引导；失败回落 v3 积木路径。"""
        r = self._add_fit_marginal(X, y, silent=False)
        if r is not None and r[3] >= 0.99:
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
        return super()._core_additive(X, y)

    # ============ 积木按需生成（替代 v3 全量预造） ============
    def _add_fit(self, X, y, silent=False, fast=False):
        """Layered on-demand: single-column linear -> 2 columns (fast path)
        -> 3 columns (Surface fit).
        fast=True: only the deterministic fast path (no Surface), for
        partial-peel screening.
        Returns (blocks, allsel, coef, r2v), same as v3."""
        d = X.shape[1]
        blocks, Zcols = [], []
        dbg = getattr(self, '_dbg_fit', False)

        def _fstr(b):
            parts = []
            for i, (kind, val) in b:
                if kind == 'p':
                    parts.append('x%d%s' % (i, '' if val == 1 else ('^%d' % val)))
                else:
                    parts.append('(1-%sx%d)' % (val, i))
            return '*'.join(parts)

        def _show(tag):
            if dbg:
                print('   [%s] 选用=%d r2=%.4f : %s'
                      % (tag, len(selected), r2cur(),
                         ' | '.join(_fstr(blocks[i]) for i in selected)),
                      flush=True)

        def add_block(b):
            z = ev_block(b, X)
            if np.all(np.isfinite(z)) and np.std(z) > 1e-9:
                blocks.append(b); Zcols.append(z)
                return len(blocks) - 1
            return None

        def Zm():
            return np.column_stack(Zcols)

        # 初始：单列线性
        for k in range(d):
            add_block([(k, ('p', 1))])
        selected = []
        pool = _factor_pool(d, X)

        def resid():
            if not selected:
                return y.copy()
            Zs = np.column_stack([Zcols[i] for i in selected])
            A = np.column_stack([Zs, np.ones(len(y))])
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            return y - Zs @ coef[:-1] - coef[-1]

        r = resid()

        def r2cur():
            return SC.r2(y, y - r) if selected else -1.0

        def _gen_for(M):
            """紧三元组（3列全量）＋ 强边对（2列全量）；都无 → 单列变换。"""
            import itertools
            made = []
            triples = [t for t in itertools.combinations(range(d), 3)
                       if min(abs(M[t[0], t[1]]), abs(M[t[0], t[2]]),
                              abs(M[t[1], t[2]])) >= 0.5]
            for t in triples:
                for comb in itertools.product(*[pool[v] for v in t]):
                    bi = add_block(list(comb))
                    if bi is not None:
                        made.append(bi)
            pairs = [p for p in itertools.combinations(range(d), 2)
                     if abs(M[p[0], p[1]]) >= 0.5]
            for p in pairs:
                for comb in itertools.product(*[pool[v] for v in p]):
                    bi = add_block(list(comb))
                    if bi is not None:
                        made.append(bi)
            if not triples and not pairs:
                for k in range(d):
                    for fac in pool[k]:
                        if fac == ('p', 1):
                            continue
                        bi = add_block([(k, fac)])
                        if bi is not None:
                            made.append(bi)
            return made

        import itertools

        if fast:
            # 快速路径（部分剥离筛选）：单列 → 2列，无 Surface
            n0 = len(blocks)
            for p in itertools.combinations(range(d), 2):
                for comb in itertools.product(*[pool[v] for v in p]):
                    add_block(list(comb))
            sel1 = self._fwd_on(Zm(), y, list(range(d)), blocks=blocks,
                                silent=silent)
            selected.extend(sel1)
            if sel1:
                Zs1 = np.column_stack([Zcols[i] for i in sel1])
                A = np.column_stack([Zs1, np.ones(len(y))])
                cf, *_ = np.linalg.lstsq(A, y, rcond=None)
                r = y - Zs1 @ cf[:-1] - cf[-1]
            sel2 = self._fwd_on(Zm(), r, list(range(n0, len(blocks))),
                                blocks=blocks, silent=silent)
            selected.extend(sel2)
            r = resid()
            _show('快速通道')
        else:
            # Full path (core_additive): Surface interaction -> 3+2 columns
            # on demand -> iterative fit
            for it in range(4):
                if r2cur() >= 0.9999 or len(blocks) >= 3000:
                    break
                S = Surface(X, r, epochs=PB.PROBE_EPOCHS)
                M = PB._interaction_matrix(S, X, list(range(d)))
                n0 = len(blocks)
                newc = sorted(set(c for c in _gen_for(M) if c >= n0))
                if not newc:
                    break
                # 第一轮同时让单列（E、U）参与
                cand = list(range(d)) + newc if it == 0 else newc
                sel = self._fwd_on(Zm(), r, cand, blocks=blocks,
                                   vote_th=0.35, silent=silent)
                if not sel:
                    break
                selected.extend(sel)
                r = resid()
                _show('iter-%d' % it)

        if not selected:
            return None
        Zs = np.column_stack([Zcols[i] for i in selected])
        A = np.column_stack([Zs, np.ones(len(y))])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = Zs @ coef[:-1] + coef[-1]
        r2v = SC.r2(y, pred)
        if not silent:
            self.log.append("按需积木 总=%d 选用=%d r2=%.4f"
                            % (len(blocks), len(selected), r2v))
        self.last_nblocks = len(blocks)
        return blocks, selected, coef, r2v

    def _solve_r(self, X, y, col_expr, names, depth):
        if depth > 4:
            return None, None, None
        Xcur = X  # 本层输入（pred 基准）
        ce_local = [('var', i) for i in range(X.shape[1])]
        nm_local = list(names)
        col_expr = list(col_expr)
        # 前置剥层（X 变为剥层后 Xpe）
        X, y, col_expr, ce_local, nm_local, added, added_local = \
            self._peel_local(X, y, col_expr, ce_local, nm_local)
        # 部分剥离（在 Xpe 上改 y，不删列）
        for _ in range(3):
            Xn, yn, pa, pa_loc = self._peel_partial_local(
                X, y, col_expr, ce_local, nm_local)
            if not pa:
                break
            y = yn; added.extend(pa); added_local.extend(pa_loc)

        def _finish(core_e):
            e_loc = self._restore_added(_lift(core_e, ce_local), added_local)
            e_X0 = self._restore_added(_lift(core_e, col_expr), added)
            return e_X0, e_loc, _ev_c(e_loc, Xcur)

        if X.shape[1] == 0:
            ce = ('const', float(np.median(y)))
            e_loc = self._restore_added(ce, added_local)
            e_X0 = self._restore_added(ce, added)
            return e_X0, e_loc, _ev_c(e_loc, Xcur)
        # outer hint
        st = PB.State(X, y, ce_local, nm_local)
        mi = PB.aggregate_observations(PB.scan_all(st))
        hint = None
        for c, evd, det in mi['outer']:
            if evd:
                hint = evd['wrap']; break
        colidx = list(range(X.shape[1]))
        # 1) 一元组合
        core = self._core_unary(X, y, colidx, nm_local)
        if core is not None:
            return _finish(core[1])
        # 2) 加性（高精度）
        add = self._core_additive(X, y)
        if add is not None and add[1] >= 0.99:
            return _finish(add[0])
        # 3) 外层递归（用内层 pred 验证）
        order = ['sqrt', 'exp']
        if hint:
            order = [hint] + [w for w in ['sqrt', 'exp'] if w != hint]
        for w in order:
            if w == 'sqrt' and (np.all(y > 0) or np.all(y < 0)):
                z = y ** 2
                ceX0 = list(col_expr); nmX = list(nm_local)
                iX0, i_loc, pred_z = self._solve_r(
                    X.copy(), z, ceX0, nmX, depth + 1)
                if iX0 is not None and SC.r2(z, pred_z) > 0.99:
                    return _finish(('sqrt', i_loc))
            if w == 'exp' and np.all(y > 0):
                z = np.log(np.clip(y, 1e-9, None))
                ceX0 = list(col_expr); nmX = list(nm_local)
                iX0, i_loc, pred_z = self._solve_r(
                    X.copy(), z, ceX0, nmX, depth + 1)
                if iX0 is not None and SC.r2(z, pred_z) > 0.99:
                    return _finish(('exp', i_loc))
        # 加性兜底
        if add is not None:
            return _finish(add[0])
        return None, None, None


def auto_decompose_v4(X, y, names, logy=None):
    return SHRv4(X, y, names, logy=logy).run()
