# -*- coding: utf-8 -*-
"""SHR v5 auto-decomposition: peel-and-revalidate, coupling/decoupling,
sin/cos angles, and the full probe set.

Changes over v4:
  1. Multiplicative peeling: high confidence (c>=PEEL_HARD) peels directly;
     medium confidence (PEEL_TRY<=c<PEEL_HARD) peels tentatively and
     recurses, confirmed only when out-of-sample r2>=0.99, else rolled back.
  2. Coupling/decoupling: after peeling, difference/product/ratio/ternary
     composite blocks are built (rules as in features_common) with
     expression-tree consistency checks; bin_r2 selects the top ones to
     identify composite angles/variables.
  3. sin/cos angles: tried on columns (including composite angles),
     filtered by bin_r2, keeping only those that explain the residual.
The solver layer reuses v4 unary/additive handling (margin guidance finds
p=1/p=2 on columns containing sin/cos).

Expression references follow v4: col_expr points at the original X0 and
ce_local at the current layer; peeling updates both.
"""
import numpy as np
from itertools import combinations
import probes as PB
import shr_collapse as SC
from auto_decompose_v4 import SHRv4, _lift, MARG_ETA, MARG_VOTE
from auto_decompose_v3 import block_expr, ev_block

# ---------- 旋钮 ----------
PEEL_HARD = 0.90   # 高置信直接剥
PEEL_TRY = 0.30    # 中置信试剥
COUPLE_MIN = 0.12  # 三元积木对目标 bin_r2 下限
COUPLE_TOP = 10    # 三元积木每轮最多保留
PAIR_CAP = 36      # 两两积木（差/积/比）全保留上限
TRIG_MIN = 0.15    # sin/cos 对目标 bin_r2 下限
TRIG_TOP = 8       # sin/cos 每轮最多保留
CROSS_MIN = 0.12   # 振幅×三角 二阶合成 bin_r2 下限
CROSS_TOP = 200      # 二阶合成每轮最多保留（放开，交前向选择）
ANCHOR_ROUNDS = 3  # 耦合解耦+sin/cos 迭代轮数


def _br(z, y):
    return float(np.mean([SC.bin_r2(z, y, nb) for nb in (10, 15)]))


def _sinc_z(z):
    """稳健 sinc：sin(z)/z，z=0 处取 1；小负 z 不能 clip 到 1e-9。"""
    zd = np.where(np.abs(z) < 1e-9, 1.0, z)
    return np.where(np.abs(z) < 1e-6, 1.0, np.sin(zd) / zd)


def _ev_c(e, X):
    """在 X 上求值；v5 扩展支持 sin/cos（v4 缺这两个分支）。"""
    t = e[0]
    if t in ('c', 'var'):
        return X[:, e[1]]
    if t == 'const':
        return np.full(len(X), e[1])
    if t == 'pow':
        base = _ev_c(e[1], X)
        p = e[2]
        # 整数幂允许负底（如 cos²，负 cos 平方仍正确）；
        # 非整数幂才 clip 正底（如 x^0.5）
        if float(p).is_integer():
            return base ** p
        return np.clip(base, 1e-7, None) ** p
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
    if t == 'sin':
        return np.sin(np.clip(_ev_c(e[1], X), -50, 50))
    if t == 'cos':
        return np.cos(np.clip(_ev_c(e[1], X), -50, 50))
    if t == 'inv1m':
        return 1.0 / (1.0 - _ev_c(e[1], X))
    if t == 'sinc':
        # 稳健 sinc：sin(z)/z，z=0 处取 1；小负 z 不能 clip 到 1e-9
        z = _ev_c(e[1], X)
        zd = np.where(np.abs(z) < 1e-9, 1.0, z)
        return np.where(np.abs(z) < 1e-6, 1.0, np.sin(zd) / zd)
    if t in ('asin', 'arcsin'):
        return np.arcsin(np.clip(_ev_c(e[1], X), -1.0, 1.0))
    if t in ('acos', 'arccos'):
        return np.arccos(np.clip(_ev_c(e[1], X), -1.0, 1.0))
    if t in ('atan', 'arctan'):
        return np.arctan(_ev_c(e[1], X))
    raise ValueError(t)


class SHRv5(SHRv4):

    def run(self):
        col_expr = [('var', i) for i in range(self.d0)]
        e = self._solve(self.X0.copy(), self.y0.copy(), col_expr,
                        list(self.names0), 0)
        if e is None:
            return None, float('nan'), self.log
        e = self._lift(e, col_expr)
        pred = _ev_c(e, self.X0)
        r2v = SC.r2(self.y0, pred)
        return e, r2v, self.log

    # ============================================================
    # 1) 剥乘性：高置信直接剥 + 中置信试剥-复验
    # ============================================================
    def _peel_v5(self, X, y, col_expr, ce_local, nm_local, depth):
        added, added_local = [], []

        def mult_cands(X, y, ce_local, nm_local):
            st = PB.State(X, y, ce_local, nm_local)
            mi = PB.aggregate_observations(PB.scan_all(st, probes=[PB.MultProbe]))
            out = []
            for k, (c, det) in mi['mult'].items():
                p = det.get('power') if det else None
                if p is None or round(p) == 0 or abs(p - round(p)) > 1e-9:
                    continue
                out.append((c, k, int(round(p))))
            return out

        # 统一试剥-复验：MultProbe 会把"加性项里的乘性变量"（如
        # I_12_11 的 B，只出现在 q·B·v·sin 这一项）误判为整体乘性，
        # 即使 c=1.00 也不能直接确认。因此所有候选都要"剥后递归 solve、
        # 样本外 r2>=0.99 才确认"，否则回退；按置信度从高到低逐个试，
        # 第一个复验通过的确认剥。每剥一列列数递减，递归必终止。
        for _ in range(10):
            mc = [t for t in mult_cands(X, y, ce_local, nm_local)
                  if t[0] >= PEEL_TRY]
            if not mc:
                break
            mc.sort(reverse=True)
            confirmed = False
            for c, k, p in mc[:4]:   # 同层最多试 4 个候选，防乱试
                nmk = nm_local[k]
                keep = [i for i in range(X.shape[1]) if i != k]
                yt = y / np.power(np.clip(X[:, k], 1e-9, None), p)
                if not np.all(np.isfinite(yt)):
                    continue
                Xt = X[:, keep] if keep else np.empty((len(y), 0))
                ce0_t = [col_expr[i] for i in keep]
                nmL_t = [nm_local[i] for i in keep]
                # 递归复验：走对外入口，ce_local 相对 Xt 干净重建
                iX0, iLoc, pred_t = self._solve_r(Xt, yt, ce0_t,
                                                  nmL_t, depth + 1)
                rr = SC.r2(yt, pred_t) if iX0 is not None else float('nan')
                if iX0 is not None and rr >= 0.99:
                    added.append(('mult', col_expr[k], p))
                    added_local.append(('mult', ce_local[k], p))
                    X, y = Xt, yt
                    col_expr[:] = ce0_t
                    ce_local[:] = [ce_local[i] for i in keep]
                    nm_local[:] = nmL_t
                    confirmed = True
                    self.log.append("v5剥 %s^%s (c=%.2f) 复验r2=%.4f" % (
                        nmk, p, c, rr))
                    break
                # 复验失败：回退，试下一个候选
            if not confirmed:
                break
        return X, y, col_expr, ce_local, nm_local, added, added_local

    # ============================================================
    # 2) 耦合解耦：复合积木（数值+表达式树同步，含一致性校验）
    # ============================================================
    def _compound_blocks(self, X, Xcur, nbase, col_expr, ce_local):
        d = nbase
        pairwise, triple = [], []

        def add(lst, name, col, e0, eL):
            if not np.all(np.isfinite(col)) or np.max(np.abs(col)) > 1e6:
                return
            # 一致性校验在 Xcur（eL 相对 Xcur），不是在剥后 X
            chk = _ev_c(eL, Xcur)
            if np.allclose(chk, col, rtol=1e-5, atol=1e-7):
                lst.append((name, col, e0, eL))

        for a in range(d):
            for b in range(a + 1, d):
                add(pairwise, "d%d%d" % (a + 1, b + 1), X[:, a] - X[:, b],
                    ('sub', col_expr[a], col_expr[b]),
                    ('sub', ce_local[a], ce_local[b]))
                add(pairwise, "p%d%d" % (a + 1, b + 1), X[:, a] * X[:, b],
                    ('mul', col_expr[a], col_expr[b]),
                    ('mul', ce_local[a], ce_local[b]))
                add(pairwise, "r%d%d" % (a + 1, b + 1),
                    X[:, a] / np.clip(X[:, b], 1e-9, None),
                    ('div', col_expr[a], col_expr[b]),
                    ('div', ce_local[a], ce_local[b]))
                add(pairwise, "r%d%d" % (b + 1, a + 1),
                    X[:, b] / np.clip(X[:, a], 1e-9, None),
                    ('div', col_expr[b], col_expr[a]),
                    ('div', ce_local[b], ce_local[a]))
        for a, b, c in combinations(range(d), 3):
            add(triple, "tpp%d%d%d" % (a + 1, b + 1, c + 1),
                X[:, a] * X[:, b] * X[:, c],
                ('mul', ('mul', col_expr[a], col_expr[b]), col_expr[c]),
                ('mul', ('mul', ce_local[a], ce_local[b]), ce_local[c]))
            for x, yz, zt in [
                (a, b, c), (a, c, b), (b, c, a)]:
                add(triple, "ppd%d%d_%d" % (x + 1, yz + 1, zt + 1),
                    X[:, x] * X[:, yz] / np.clip(X[:, zt], 1e-9, None),
                    ('div', ('mul', col_expr[x], col_expr[yz]), col_expr[zt]),
                    ('div', ('mul', ce_local[x], ce_local[yz]), ce_local[zt]))
                add(triple, "dpx%d%d_%d" % (x + 1, yz + 1, zt + 1),
                    (X[:, x] - X[:, yz]) * X[:, zt],
                    ('mul', ('sub', col_expr[x], col_expr[yz]), col_expr[zt]),
                    ('mul', ('sub', ce_local[x], ce_local[yz]), ce_local[zt]))
        return pairwise, triple

    def _couple_decouple(self, X, Xcur, nbase, y, col_expr, ce_local, nm_local):
        pairwise, triple = self._compound_blocks(
            X, Xcur, nbase, col_expr, ce_local)

        def _skey(t):
            s = t[0]
            return -(s if np.isfinite(s) else -1.0)

        # 两两积木（差/积/比）：中间产物（如 B·v 振幅）单独对 y 可能不
        # 相关，被严尺过滤会导致后续 cross 造不出 B·v·sin，因此全保留
        # （数量 = C(d,2)×4，d 小；加 cap 防 d 过大）
        kept = pairwise[:PAIR_CAP]
        # 三元积木数量多：按对 y 的 bin_r2 筛选 top
        scored = [(_br(col, y), name, col, e0, eL)
                  for name, col, e0, eL in triple]
        scored.sort(key=_skey)
        kept += [(t[1], t[2], t[3], t[4])
                 for t in scored if t[0] >= COUPLE_MIN][:COUPLE_TOP]
        if not kept:
            return X, col_expr, ce_local, nm_local, False
        X = np.column_stack([X] + [t[1] for t in kept])
        col_expr.extend([t[2] for t in kept])
        ce_local.extend([t[3] for t in kept])
        nm_local.extend([t[0] for t in kept])
        self.log.append("v5耦合解耦 加%d: %s" % (
            len(kept), [t[0] for t in kept]))
        return X, col_expr, ce_local, nm_local, True

    # ============================================================
    # 3) 角 sin/cos
    # ============================================================
    def _add_trig(self, X, Xcur, trig_base, y, col_expr, ce_local, nm_local):
        d = trig_base
        cands = []
        for k in range(d):
            u = X[:, k]
            for fn, tag in ((np.sin, 'sin'), (np.cos, 'cos')):
                col = fn(np.clip(u, -50, 50))
                eL = (tag, ce_local[k])
                e0 = (tag, col_expr[k])
                # 一致性校验在 Xcur（eL 相对 Xcur）
                chk = _ev_c(eL, Xcur)
                if not np.allclose(chk, col, rtol=1e-5, atol=1e-7):
                    continue
                s = _br(col, y)
                if s >= TRIG_MIN:
                    cands.append((s, tag + "(" + nm_local[k] + ")", col,
                                  e0, eL))
        cands.sort(key=lambda t: -(t[0] if np.isfinite(t[0]) else -1.0))
        cands = cands[:TRIG_TOP]
        if not cands:
            return X, col_expr, ce_local, nm_local, False
        X = np.column_stack([X] + [t[2] for t in cands])
        col_expr.extend([t[3] for t in cands])
        ce_local.extend([t[4] for t in cands])
        nm_local.extend([t[1] for t in cands])
        self.log.append("v5加三角 加%d: %s" % (
            len(cands), [t[1] for t in cands]))
        return X, col_expr, ce_local, nm_local, True

    # ============================================================
    # 4) 二阶合成：振幅列（原始+复合）× 三角列
    #    如 I_12_11 的 B·v·sin(theta)、III_17_37 的 alpha·cos(theta)。
    #    只在一层列上做，数量 = 非三角列 × 三角列，有界（不嵌套、不爆炸）。
    # ============================================================
    def _cross_trig(self, X, Xcur, trig_base, y, col_expr, ce_local,
                    nm_local):
        total = X.shape[1]
        cands = []
        for i in range(trig_base):              # 振幅列（原始+复合）
            for j in range(trig_base, total):  # 三角列
                col = X[:, i] * X[:, j]
                if not np.all(np.isfinite(col)) or np.max(np.abs(col)) > 1e6:
                    continue
                eL = ('mul', ce_local[i], ce_local[j])
                e0 = ('mul', col_expr[i], col_expr[j])
                chk = _ev_c(eL, Xcur)
                if not np.allclose(chk, col, rtol=1e-5, atol=1e-7):
                    continue
                s = _br(col, y)
                if s >= CROSS_MIN:
                    cands.append((s, nm_local[i] + "*" + nm_local[j], col,
                                  e0, eL))
        cands.sort(key=lambda t: -(t[0] if np.isfinite(t[0]) else -1.0))
        cands = cands[:CROSS_TOP]
        if not cands:
            return X, col_expr, ce_local, nm_local, False
        X = np.column_stack([X] + [t[2] for t in cands])
        col_expr.extend([t[3] for t in cands])
        ce_local.extend([t[4] for t in cands])
        nm_local.extend([t[1] for t in cands])
        self.log.append("v5二阶合成 加%d: %s" % (
            len(cands), [t[1] for t in cands]))
        return X, col_expr, ce_local, nm_local, True

    # ============================================================
    # 5) 一次构造全部列（在原始 X 上，记录 deps 原始变量依赖集）
    # ============================================================
    def _build_all(self, X, y, col_expr):
        """在原始 X 上一次性构造：原始 → pairwise/triple → sin/cos →
        振幅×三角。返回 (X_ext, col_expr_ext, ce_ext, nm_ext, deps)。
        ce 相对 X（本层），col_expr 相对 X0（对外）。"""
        n = X.shape[1]
        vcols = [X[:, i] for i in range(n)]
        ce = [('var', i) for i in range(n)]
        nm = [col_expr[i] if isinstance(col_expr[i], str) else
              ("x%d" % i) for i in range(n)]
        cex = [col_expr[i] for i in range(n)]
        deps = [frozenset([i]) for i in range(n)]

        def addc(col, dep, eL, e0, name):
            if not np.all(np.isfinite(col)) or np.max(np.abs(col)) > 1e6:
                return
            chk = _ev_c(eL, X)
            if np.allclose(chk, col, rtol=1e-5, atol=1e-7):
                vcols.append(col); deps.append(dep)
                ce.append(eL); cex.append(e0); nm.append(name)

        # 5.1 pairwise（差/积/双向比），全保留
        pair = []
        for a in range(n):
            for b in range(a + 1, n):
                pair.append((X[:, a] - X[:, b], frozenset([a, b]),
                             ('sub', ('var', a), ('var', b)),
                             ('sub', col_expr[a], col_expr[b]),
                             "d%d%d" % (a + 1, b + 1)))
                pair.append((X[:, a] * X[:, b], frozenset([a, b]),
                             ('mul', ('var', a), ('var', b)),
                             ('mul', col_expr[a], col_expr[b]),
                             "p%d%d" % (a + 1, b + 1)))
                pair.append((X[:, a] / np.clip(X[:, b], 1e-9, None),
                             frozenset([a, b]),
                             ('div', ('var', a), ('var', b)),
                             ('div', col_expr[a], col_expr[b]),
                             "r%d%d" % (a + 1, b + 1)))
                pair.append((X[:, b] / np.clip(X[:, a], 1e-9, None),
                             frozenset([a, b]),
                             ('div', ('var', b), ('var', a)),
                             ('div', col_expr[b], col_expr[a]),
                             "r%d%d" % (b + 1, a + 1)))
        pair = pair[:PAIR_CAP]
        for t in pair:
            addc(*t)

        # 5.1b pairwise 差平方（几何确定、全保留；数量 C(n,2)，
        # d≤7 时 ≤21，有界）。余弦定理 sin 半角形式、距离公式等都依赖它；
        # 单列 r2 不高但作为"第二项"必不可少，不受 PAIR_CAP 截断。
        for a in range(n):
            for b in range(a + 1, n):
                d_ab = X[:, a] - X[:, b]
                dL = ('sub', ('var', a), ('var', b))
                d0 = ('sub', col_expr[a], col_expr[b])
                addc(d_ab ** 2, frozenset([a, b]),
                     ('pow', dL, 2), ('pow', d0, 2),
                     "d2%d%d" % (a + 1, b + 1))

        # 5.1c pairwise 和 + 平方差（几何确定、全保留，数量 C(n,2) 有界）。
        # 和：总质量/总量复合；平方差 a²-b² 与差平方 (a-b)² 不同，
        # test_12 的 y²-d²、因式分解等依赖它，不构造则加性列空间缺该项。
        for a in range(n):
            for b in range(a + 1, n):
                s_ab = X[:, a] + X[:, b]
                addc(s_ab, frozenset([a, b]),
                     ('add', ('var', a), ('var', b)),
                     ('add', col_expr[a], col_expr[b]),
                     "s%d%d" % (a + 1, b + 1))
                sd = X[:, a] ** 2 - X[:, b] ** 2
                addc(sd, frozenset([a, b]),
                     ('sub', ('pow', ('var', a), 2), ('pow', ('var', b), 2)),
                     ('sub', ('pow', col_expr[a], 2), ('pow', col_expr[b], 2)),
                     "sd2%d%d" % (a + 1, b + 1))

        # 5.2 triple（积/ppd/dpx），按 bin_r2 选 top
        tri = []
        for a, b, c in combinations(range(n), 3):
            tri.append((X[:, a] * X[:, b] * X[:, c], frozenset([a, b, c]),
                        ('mul', ('mul', ('var', a), ('var', b)), ('var', c)),
                        ('mul', ('mul', col_expr[a], col_expr[b]), col_expr[c]),
                        "tpp%d%d%d" % (a + 1, b + 1, c + 1)))
            for x, yz, zt in [(a, b, c), (a, c, b), (b, c, a)]:
                tri.append((X[:, x] * X[:, yz] /
                            np.clip(X[:, zt], 1e-9, None),
                            frozenset([a, b, c]),
                            ('div', ('mul', ('var', x), ('var', yz)),
                             ('var', zt)),
                            ('div', ('mul', col_expr[x], col_expr[yz]),
                             col_expr[zt]),
                            "ppd%d%d_%d" % (x + 1, yz + 1, zt + 1)))
                tri.append(((X[:, x] - X[:, yz]) * X[:, zt],
                            frozenset([a, b, c]),
                            ('mul', ('sub', ('var', x), ('var', yz)),
                             ('var', zt)),
                            ('mul', ('sub', col_expr[x], col_expr[yz]),
                             col_expr[zt]),
                            "dpx%d%d_%d" % (x + 1, yz + 1, zt + 1)))
        scored = [(_br(t[0], y), t) for t in tri]
        scored.sort(key=lambda t: -(t[0] if np.isfinite(t[0]) else -1))
        for s, t in scored:
            if s >= COUPLE_MIN:
                addc(*t)
        if len(vcols) > n + len(pair) + COUPLE_TOP:
            pass

        # 5.3 sin/cos（在原始+pairwise+triple 列上）
        trig_base = len(vcols)
        tg = []
        tg_keep = []
        for k in range(trig_base):
            for fn, tag in ((np.sin, 'sin'), (np.cos, 'cos')):
                col = fn(np.clip(vcols[k], -50, 50))
                eL = (tag, ce[k]); e0 = (tag, cex[k])
                chk = _ev_c(eL, X)
                if not np.allclose(chk, col, rtol=1e-5, atol=1e-7):
                    continue
                s = _br(col, y)
                if s >= TRIG_MIN:
                    rec = (s, col, deps[k], eL, e0,
                           tag + "(" + nm[k] + ")")
                    if k < n:
                        # 原始变量的 sin/cos 互补、必须成对保留（否则
                        # cos(theta)·sin(theta) 这类缺一个就构造不出）
                        tg_keep.append(rec)
                    else:
                        tg.append(rec)
        tg.sort(key=lambda t: -t[0])
        tg = tg[:TRIG_TOP]
        for s, col, dep, eL, e0, name in tg_keep + tg:
            addc(col, dep, eL, e0, name)

        # 5.3b 每个原始角变量的确定性三角积木：sin·cos、sin²、cos²
        # （这些是几何确定结构，不做全局 bin_r2 截断——否则会像
        # cos(theta)·sin(theta) 那样被"混杂但单列相关性更高"的组合挤掉）。
        for k in range(n):
            u = np.clip(X[:, k], -50, 50)
            s = np.sin(u); c = np.cos(u)
            sN = nm[k]
            extras = [
                (s * c, frozenset([k]),
                 ('mul', ('sin', ('var', k)), ('cos', ('var', k))),
                 ('mul', ('sin', cex[k]), ('cos', cex[k])),
                 "sin*cos(" + sN + ")"),
                (s * s, frozenset([k]),
                 ('pow', ('sin', ('var', k)), 2),
                 ('pow', ('sin', cex[k]), 2),
                 "sin2(" + sN + ")"),
                (c * c, frozenset([k]),
                 ('pow', ('cos', ('var', k)), 2),
                 ('pow', ('cos', cex[k]), 2),
                 "cos2(" + sN + ")"),
            ]
            for col, dep, eL, e0, name in extras:
                addc(col, dep, eL, e0, name)

        # 5.3b2 原始角角度缩放：半角 sin(theta/2)（test_1）、倍角等。
        # 5.3c 只对复合角，原始角的缩放在此补；门控（与 y 相关才加）。
        for k in range(n):
            u = X[:, k]
            for c in (0.5, np.pi, 2 * np.pi):
                cu = np.clip(c * u, -50, 50)
                s = np.sin(cu); co = np.cos(cu)
                mul_e = ('mul', ('const', float(c)), ('var', k))
                mul_0 = ('mul', ('const', float(c)), cex[k])
                forms = [
                    (s, ('sin', mul_e), ('sin', mul_0),
                     "sin(%g*%s)" % (c, nm[k])),
                    (co, ('cos', mul_e), ('cos', mul_0),
                     "cos(%g*%s)" % (c, nm[k])),
                    (s * s, ('pow', ('sin', mul_e), 2),
                     ('pow', ('sin', mul_0), 2),
                     "sin2(%g*%s)" % (c, nm[k])),
                    (co * co, ('pow', ('cos', mul_e), 2),
                     ('pow', ('cos', mul_0), 2),
                     "cos2(%g*%s)" % (c, nm[k])),
                    (s * co, ('mul', ('sin', mul_e), ('cos', mul_e)),
                     ('mul', ('sin', mul_0), ('cos', mul_0)),
                     "sincos(%g*%s)" % (c, nm[k])),
                ]
                # 倒数形式（1/sin²、1/cos²、1/(sin·cos)）：分母不趋零才加，
                # 防发散；门控（与 y 相关才加）。
                if np.min(np.abs(s)) > 1e-6:
                    forms.append(
                        (1.0 / s**2,
                         ('div', ('const', 1.0),
                          ('pow', ('sin', mul_e), 2)),
                         ('div', ('const', 1.0),
                          ('pow', ('sin', mul_0), 2)),
                         "inv_sin2(%g*%s)" % (c, nm[k])))
                if np.min(np.abs(co)) > 1e-6:
                    forms.append(
                        (1.0 / co**2,
                         ('div', ('const', 1.0),
                          ('pow', ('cos', mul_e), 2)),
                         ('div', ('const', 1.0),
                          ('pow', ('cos', mul_0), 2)),
                         "inv_cos2(%g*%s)" % (c, nm[k])))
                if np.min(np.abs(s * co)) > 1e-6:
                    forms.append(
                        (1.0 / (s * co),
                         ('div', ('const', 1.0),
                          ('mul', ('sin', mul_e), ('cos', mul_e))),
                         ('div', ('const', 1.0),
                          ('mul', ('sin', mul_0), ('cos', mul_0))),
                         "inv_sincos(%g*%s)" % (c, nm[k])))
                for col, eL, e0, name in forms:
                    if _br(col, y) >= CROSS_MIN:
                        addc(col, frozenset([k]), eL, e0, name)

        # 5.3c 角常数缩放：对复合角候选 z（sin(z)/cos(z) 与 y 相关，
        # 但角度可能缺常数），构造带常数的 sin/cos/sinc 及其二次形式。
        # 通用：物理角常含 2π（III_8_54）、半角 1/2（III_9_52）；
        # 只对"已显振荡相关"的列触发，不盲目给所有列加。
        # 数量有界（角候选列 × 3 常数 × 7 形式）。
        ANG_C = (1.0, 2 * np.pi, np.pi, 0.5)
        for k in range(n, trig_base):
            z = vcols[k]
            if not np.all(np.isfinite(z)) or np.max(np.abs(z)) > 50:
                continue
            base_s = max(_br(np.sin(np.clip(z, -50, 50)), y),
                         _br(np.cos(np.clip(z, -50, 50)), y))
            if base_s < 0.25:
                continue
            for c in ANG_C:
                cz = np.clip(c * z, -50, 50)
                s = np.sin(cz); co = np.cos(cz); sc = _sinc_z(cz)
                mul_e = ('mul', ('const', float(c)), ce[k])
                mul_0 = ('mul', ('const', float(c)), cex[k])
                forms = [
                    (s * s, ('pow', ('sin', mul_e), 2),
                     ('pow', ('sin', mul_0), 2),
                     "sin2(%g*%s)" % (c, nm[k])),
                    (co * co, ('pow', ('cos', mul_e), 2),
                     ('pow', ('cos', mul_0), 2),
                     "cos2(%g*%s)" % (c, nm[k])),
                    (sc * sc, ('pow', ('sinc', mul_e), 2),
                     ('pow', ('sinc', mul_0), 2),
                     "sinc2(%g*%s)" % (c, nm[k])),
                    (s * co, ('mul', ('sin', mul_e), ('cos', mul_e)),
                     ('mul', ('sin', mul_0), ('cos', mul_0)),
                     "sincos(%g*%s)" % (c, nm[k])),
                    (s, ('sin', mul_e), ('sin', mul_0),
                     "sin(%g*%s)" % (c, nm[k])),
                    (co, ('cos', mul_e), ('cos', mul_0),
                     "cos(%g*%s)" % (c, nm[k])),
                    (sc, ('sinc', mul_e), ('sinc', mul_0),
                     "sinc(%g*%s)" % (c, nm[k])),
                ]
                for col, eL, e0, name in forms:
                    if _br(col, y) >= CROSS_MIN:
                        addc(col, deps[k], eL, e0, name)

        # 5.4 振幅（非三角列）× 三角列
        trig_cols = list(range(trig_base, len(vcols)))
        cr = []
        for i in range(trig_base):
            for j in trig_cols:
                if deps[i] & deps[j]:
                    continue
                col = vcols[i] * vcols[j]
                eL = ('mul', ce[i], ce[j]); e0 = ('mul', cex[i], cex[j])
                chk = _ev_c(eL, X)
                if not np.allclose(chk, col, rtol=1e-5, atol=1e-7):
                    continue
                s = _br(col, y)
                if s >= CROSS_MIN:
                    cr.append((s, col, deps[i] | deps[j], eL, e0,
                               nm[i] + "*" + nm[j]))
        cr.sort(key=lambda t: -t[0])
        cr = cr[:CROSS_TOP]
        for s, col, dep, eL, e0, name in cr:
            addc(col, dep, eL, e0, name)

        # 5.5 三角×三角（同角 cos(theta)·sin(theta) 等，如 II_6_15b）
        # 注意：两个三角 deps 相同（同一角），是合法结构，不做 deps 重叠过滤；
        # 数量 = 三角列两两，有界（trig_cols ≤ TRIG_TOP）。
        ct2 = []
        for i, j in combinations(trig_cols, 2):
            if deps[i] != deps[j]:
                continue
            col = vcols[i] * vcols[j]
            if not np.all(np.isfinite(col)) or np.max(np.abs(col)) > 1e6:
                continue
            eL = ('mul', ce[i], ce[j]); e0 = ('mul', cex[i], cex[j])
            chk = _ev_c(eL, X)
            if not np.allclose(chk, col, rtol=1e-5, atol=1e-7):
                continue
            s = _br(col, y)
            if s >= CROSS_MIN:
                ct2.append((s, col, deps[i] | deps[j], eL, e0,
                            nm[i] + "*" + nm[j]))
        ct2.sort(key=lambda t: -t[0])
        ct2 = ct2[:CROSS_TOP]
        for s, col, dep, eL, e0, name in ct2:
            addc(col, dep, eL, e0, name)

        X_ext = np.column_stack(vcols)
        return X_ext, cex, ce, nm, deps

    # ============================================================
    # 列引用提升：在 v3 基础上补 sin/cos
    # ============================================================
    @staticmethod
    def _lift(e, col_expr):
        t = e[0]
        if t == 'c':
            return col_expr[e[1]]
        if t in ('var', 'const'):
            return e
        if t == 'pow':
            return ('pow', SHRv5._lift(e[1], col_expr), e[2])
        if t in ('exp', 'sqrt', 'log', 'inv1m', 'sin', 'cos'):
            return (t, SHRv5._lift(e[1], col_expr))
        return (t, SHRv5._lift(e[1], col_expr), SHRv5._lift(e[2], col_expr))

    # ============================================================
    # 加性求解：只走边际引导（有界），不回落到 v3 全量（在合成列上爆炸）
    # ============================================================
    @staticmethod
    def _pow_zc(centers, p):
        """整数幂对 bin 中心，保留符号（v4 用 clip 到正数，负底数半边丢）。"""
        if p >= 1:
            return centers ** p
        a = np.clip(np.abs(centers), 1e-9, None)
        return np.sign(centers) * a ** p

    def _fit_marginal_forms(self, centers, mu):
        """候选 fac 对边际的拟合 r2，返回排序 [(r2, fac)]。
        v5 覆盖：整数幂用 sign-aware _pow_zc（含负底数），修复
        '振幅×三角'（有正有负，如 B·v·sin）列的线性/p 识别。
        简洁性 tie-break：p=1（直接用列）与 s1/s2（列的线性变换）在"线性
        关系"上 r2 本应相同（列空间一样），数值噪声下排序不稳；在 r2 差
        <0.005 的同档里选复杂度最低的 fac（奥卡姆，防过拟合到假 fac）。"""
        res = []
        for p in (-4, -3, -2, -1, 1, 2, 3, 4):
            zc = self._pow_zc(centers, p)
            if np.std(zc) < 1e-9:
                continue
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
        res.sort(key=lambda x: -x[0])
        COMP = {('p', 1): 0, ('s', 1): 1, ('p', 2): 2, ('s', 2): 3,
                ('p', 3): 4, ('p', -1): 5, ('p', 4): 6, ('p', -2): 7,
                ('p', -3): 8, ('p', -4): 9}
        best_r2 = res[0][0]
        near = [x for x in res if x[0] >= best_r2 - 0.005]
        near.sort(key=lambda x: COMP.get(x[1], 99))
        chosen = near[0]
        if chosen != res[0]:
            res = [chosen] + [x for x in res if x != chosen]
        return res

    def _add_fit_marginal(self, X, y, keep=None, silent=False):
        """边际引导（v5）：backfitting，每轮"在当前残差上对全部列重算边际"，
        只投票选一个最佳项再投影，下一轮候选重算。
        这样大项（Bv sin，方差主导）先剥；剥后残差里干净小项（Ef）的边际
        eta 升到 1，能被选中；不会在第一轮就把候选锁死、让混杂列（Ef-theta）
        挤掉干净小项。
        keep：可用"全局列索引"子集（剥后过滤用）；blocks 里 k 用全局列索引。
        """
        if keep is None:
            keep = list(range(X.shape[1]))
        blocks, Zcols = [], []

        def add_block(b):
            hit = next((i for i in range(len(blocks)) if blocks[i] == b), None)
            if hit is not None:
                return hit
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

        for rnd in range(8):
            if r2cur() >= 0.9999:
                break
            r = resid()
            # 当前残差上对全部 keep 列重算边际 eta
            eta_list = []
            for k in keep:
                eta, centers, mu = self._marginal_profile(X[:, k], r)
                eta_list.append((eta, k, centers, mu))
            eta_list.sort(key=lambda t: -t[0])
            # 取 eta 最高的一批，拟合各自 fac
            cand = []
            for eta, k, centers, mu in eta_list[:10]:
                if eta < 0.03:
                    continue
                forms = self._fit_marginal_forms(centers, mu)
                r2f, fac = forms[0]
                bi = add_block([(k, fac)])
                if bi is not None and bi not in selected:
                    cand.append(bi)
            if not cand:
                if not silent:
                    print("      marg rnd%d: 无候选 (r2cur=%.4f)" % (
                        rnd, r2cur()), flush=True)
                break
            # 投票只选一个（max_steps=1），下一轮重算候选
            sel = self._fwd_on(np.column_stack(Zcols), r, cand,
                               blocks=blocks, vote_th=MARG_VOTE,
                               max_steps=1, silent=True)
            if not silent:
                print("      marg rnd%d: cand=%d 新增sel=%s r2cur=%.4f" % (
                    rnd, len(cand),
                    [block_expr(blocks[i]) for i in sel],
                    r2cur()), flush=True)
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
        r = self._add_fit_marginal(X, y, silent=False)
        if r is None or r[3] < 0.99:
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

    # ============================================================
    # 主求解
    # ============================================================
    def _peel_inspace(self, Xorig, X_ext, y, deps, active, depth):
        """在"已构造好的列空间"上剥乘性。
        判据（剥后 r2）：先在当前 active 上不剥解 r0；再对每个候选剥后解一次
        （不递归，廉价）。
          - 剥后 r2≥0.9995 → 剥后是纯加性，直接确认（剥后 r2 最高者优先）；
          - 剥后 r2 在 0.99–0.9995 → 剥后可能还有乘性，再递归；
          - 剥后 r2<0.99 → 剥错（剥后无法表达），直接拒绝（不深递归）。
        这样 c 并列（如 q、B 都≈0.998）时，靠剥后 r2 区分（q→1.0，B→0.95），
        错误候选不进深度递归。
        返回 (blocks, selected, coef, r2, peels)；blocks 用全局列索引。
        """
        keep = [j for j in range(X_ext.shape[1]) if deps[j] <= active]
        r0 = self._add_fit_marginal(X_ext, y, keep=keep, silent=True) \
            if keep else None
        r0v = r0[3] if r0 is not None else -1.0
        if r0v >= 0.9995:
            return r0[0], r0[1], r0[2], r0[3], []
        if depth > 5:
            if r0v >= 0.99:
                return r0[0], r0[1], r0[2], r0[3], []
            return None
        act_list = sorted(active)
        Xa = Xorig[:, act_list]
        ce_a = [('var', i) for i in range(len(act_list))]
        st = PB.State(Xa, y, ce_a, ["x%d" % i for i in act_list])
        mi = PB.aggregate_observations(PB.scan_all(st, probes=[PB.MultProbe]))
        raw = []
        for lk, (c, det) in mi['mult'].items():
            p = det.get('power') if det else None
            if p is not None:
                raw.append((c, act_list[lk], p))
        # 对每个候选剥后解一次（不递归）
        scored = []
        for c, orig_k, p in raw:
            active2 = active - {orig_k}
            xb = Xorig[:, orig_k]
            yt = y / np.power(np.clip(xb, 1e-9, None), p)
            if not np.all(np.isfinite(yt)):
                continue
            keep2 = [j for j in range(X_ext.shape[1]) if deps[j] <= active2]
            r2 = self._add_fit_marginal(X_ext, yt, keep=keep2, silent=True) \
                if keep2 else None
            v = r2[3] if r2 is not None else -1.0
            scored.append((v, orig_k, p, active2, yt, r2))
        # 按剥后 r2 降序（这才是正确的"剥谁"判据，不靠 c 排序）
        scored.sort(key=lambda t: -t[0])
        for v, orig_k, p, active2, yt, r2 in scored:
            # 剥后必须能解（≥0.99）且比不剥 r0 有收益，否则剥错
            if v < 0.99 or v <= r0v + 1e-6:
                continue
            if v >= 0.9995:
                return r2[0], r2[1], r2[2], r2[3], [(orig_k, p)]
            rec = self._peel_inspace(Xorig, X_ext, yt, deps, active2,
                                     depth + 1)
            if rec is not None and rec[3] >= 0.99:
                b, s, cf, r2v, peels = rec
                return b, s, cf, r2v, [(orig_k, p)] + peels
        # 剥都失败：回退不剥的高分近似
        if r0v >= 0.99:
            return r0[0], r0[1], r0[2], r0[3], []
        return None

    def _solve_r(self, X, y, col_expr, names, depth):
        # 对外入口（同 v4 签名）：names 为列名（字符串），内部建 ce_local
        ce_local = [('var', i) for i in range(X.shape[1])]
        return self._solve_r5(X, y, list(col_expr), ce_local,
                              list(names), depth)

    def _solve_r5(self, X, y, col_expr, ce_local, nm_local, depth):
        if depth > 5:
            return None, None, None
        Xcur = X
        col_expr = list(col_expr)

        # 1. 一次构造全部列（原始 → pairwise/triple → sin/cos → 振幅×三角）
        X_ext, cex_ext, ce_ext, nm_ext, deps = self._build_all(X, y, col_expr)
        print("    [d=%d] built X_ext=%d (orig=%d)" % (
            depth, X_ext.shape[1], X.shape[1]), flush=True)

        # 2. 列空间 peel：先直接解（不剥），失败再试剥、过滤列复验
        active = frozenset(range(X.shape[1]))
        rec = self._peel_inspace(X, X_ext, y, deps, active, depth)

        def _assemble(rec):
            blocks, selected, coef, r2v, peels = rec
            e_l, e_0 = None, None
            for j, bi in enumerate(selected):
                cj = coef[j]
                be = block_expr(blocks[bi])          # 'c' 全局列索引
                tl = self._lift(be, ce_ext)          # 相对 Xcur
                t0 = self._lift(be, cex_ext)         # 相对 X0
                if abs(cj - 1) >= 1e-9:
                    tl = ('mul', ('const', float(cj)), tl)
                    t0 = ('mul', ('const', float(cj)), t0)
                e_l = tl if e_l is None else ('add', e_l, tl)
                e_0 = t0 if e_0 is None else ('add', e_0, t0)
            b = coef[-1]
            if abs(b) > 1e-9:
                e_l = ('add', e_l, ('const', float(b)))
                e_0 = ('add', e_0, ('const', float(b)))
            # 乘回剥掉的乘性因子（整体乘，含截距）
            for orig_k, p in peels:
                fl = ('var', orig_k) if p == 1 else \
                    ('pow', ('var', orig_k), p)
                f0 = col_expr[orig_k] if p == 1 else \
                    ('pow', col_expr[orig_k], p)
                e_l = ('mul', fl, e_l)
                e_0 = ('mul', f0, e_0)
                self.log.append("v5剥 %s^%s" % (orig_k, p))
            return e_0, e_l, _ev_c(e_l, Xcur)

        if rec is not None:
            print("    [d=%d] peel hit r2=%.4f peels=%s" % (
                depth, rec[3], rec[4]), flush=True)
            return _assemble(rec)

        # 3. 外层（sqrt/exp，试剥-复验；递归走完整入口，仅少数 wrap）
        st = PB.State(X, y, [('var', i) for i in range(X.shape[1])],
                      list(nm_local))
        mi = PB.aggregate_observations(PB.scan_all(st))
        hint = None
        for c, evd, det in mi['outer']:
            if evd:
                hint = evd['wrap']
                break
        order = ['sqrt', 'exp']
        if hint:
            order = [hint] + [w for w in ['sqrt', 'exp'] if w != hint]
        for w in order:
            if w == 'sqrt' and (np.all(y > 0) or np.all(y < 0)):
                z = y ** 2
                iX0, iLoc, pred_z = self._solve_r(
                    X.copy(), z, list(col_expr), list(nm_local), depth + 1)
                if iX0 is not None and SC.r2(z, pred_z) > 0.99:
                    return iX0, ('sqrt', iLoc), _ev_c(('sqrt', iLoc), Xcur)
            if w == 'exp' and np.all(y > 0):
                z = np.log(np.clip(y, 1e-9, None))
                iX0, iLoc, pred_z = self._solve_r(
                    X.copy(), z, list(col_expr), list(nm_local), depth + 1)
                if iX0 is not None and SC.r2(z, pred_z) > 0.99:
                    return iX0, ('exp', iLoc), _ev_c(('exp', iLoc), Xcur)

        return None, None, None


def auto_decompose_v5(X, y, names, logy=None):
    return SHRv5(X, y, names, logy=logy).run()
