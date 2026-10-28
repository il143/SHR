# -*- coding: utf-8 -*-
"""SHR v7 full version (adds three general mechanisms to the V7 exact-solve
arm, with no per-problem patches).

The three mechanisms (same rules for all problems, no problem name/truth):
  1. GP fallback: a general feature pack (C_pairs diff/product/ratio, B_bare
     originals, D_deep ppd/dpx), pyoperon multi-seed search, selected by
     val; exact solving first, GP as fallback.
  2. Aggressive path: when conservative peeling stalls, multiplicative
     factors with conf>0.8 are peeled in succession (not cut off by
     intermediate r2), trying completion at each step and verifying the
     whole path with the final r2.
  3. Re-validation feedback: multi-split voting plus residual signal;
     when borderline/unstable, expand columns and re-run GP, feeding back
     until a stable result or budget end.

fit runs on the full X/y; the internal fit/val is only for GP selection;
multi-split re-validation reports generalization. quick=True uses a small
budget (smoke), otherwise the full budget.
"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler

import shr_v7 as BASE
from shr_v7 import SHRv7, _nodes, SOLVE_R2, PEEL_MIN, MAX_PEEL
import probes as PB
import shr_collapse as SC
import features_common as FC
import geom_blocks as GB
import rational_blocks as RB
from auto_decompose_v5 import _ev_c
import mult_log_solver as MLS
import outer_sqrt_blocks as OSQ
import additive_blocks as ADDB
import algebra_blocks as ALG
import common_additive_blocks as CAB
import gaussian_blocks as GAUS
import trig_blocks as TRIG
import unified_peel as UP
import structure_guided as SG
import os as _os
import transform_solve as TSOL

# J2 分式复合反演短路开关（SHR_NO_RATIONAL=1 关闭，用于对照）
RATIONAL_ENABLED = _os.environ.get("SHR_NO_RATIONAL", "0") != "1"
# 分式短路主门（全量 r2，必须严格=1，不含真值）：确定性反演真解 r2=1.000000；
# 结构错的高拟合假解 r2 卡在 .99958~.99999（<1）。find_rational 的返回门是
# SOLVE_R2(.9995) 较宽，短路门必须收紧到严格=1。最终 exact 仍以符号等价核验复核。
RATIONAL_R2 = 1.0 - 1e-7
# 分式枚举维度上限（防组合爆炸）：build_denom_factors 因子数随 d 激增（d=5 约
# 1280、d=6-8 约 2000-5600），阶段1 全数据打分每题会从 ~500s 涨到 20-50 分钟。
# J2 全量扫描对 d>=6 即 skip（保持基线），主引擎对齐同一口径：d>=6 不跑完整
# 分式枚举，走几何（d=9 的 I_9_18 geom 命中）或回退 GP。
RATIONAL_MAX_D = 5

# ---------- GP 旋钮（全量对齐 G4） ----------
GP_SYMS = "add,sub,mul,div,constant,variable,sin,cos,exp,log,sqrt,asin,acos,tanh"
P_GP = 5
GP_NTHREADS = 2
C_TOPK = 8
D_TOPK = 5
GP_MAXEVAL = 2000000
C_TH, B_TH = 0.99, 0.97
REVIEW_CUTS = 5
REVIEW_THRESH = 0.9
MAX_REFLOW = 3


class SHRv7Full(SHRv7):

    # 外层候选（sqrt/exp/log/square）在 _solve_direct 卡死时用 unified_peel 兜底
    _unified_outer = True

    # ============================================================
    # 件① GP 穷举真算兜底
    # ============================================================
    def _gp_select_names(self, Xtr, ytr, prefixes, topk):
        """在 fit 上用通用积木 bin_r2 排序，返回选中 name（不读真值）。"""
        pool = FC.build_all(Xtr)
        scored = []
        for name, col in pool.items():
            if prefixes and not any(name.startswith(p) for p in prefixes):
                continue
            br = np.mean([SC.bin_r2(col, ytr, nb) for nb in (10, 15, 20)])
            scored.append((br, name))
        scored.sort(key=lambda t: -t[0])
        return [n for _, n in scored[:topk]]

    def _gp_pack_matrix(self, X, names, reg_map):
        if not names:
            return X
        ext = FC.eval_subset(names, X, reg_map)
        return np.column_stack([X, ext])

    def _gp_fit_one(self, X, y, seed, quick):
        from pyoperon.sklearn import SymbolicRegressor
        if quick:
            reg = SymbolicRegressor(
                allowed_symbols=GP_SYMS, population_size=250, generations=250,
                max_evaluations=50000, n_threads=2, random_state=seed)
        else:
            reg = SymbolicRegressor(
                allowed_symbols=GP_SYMS, population_size=1000,
                generations=1000, max_evaluations=GP_MAXEVAL,
                n_threads=GP_NTHREADS, random_state=seed)
        reg.fit(X, y)
        return reg

    def _gp_parallel(self, Xtr, ytr, Xva, yva, quick):
        p = P_GP if not quick else 2
        with ThreadPoolExecutor(max_workers=p) as ex:
            futs = {ex.submit(self._gp_fit_one, Xtr, ytr, s, quick): s
                    for s in range(p)}
            regs = {futs[f]: f.result() for f in futs}
        vals = {s: SC.r2(yva, regs[s].predict(Xva)) for s in regs}
        bs = max(vals, key=lambda s: vals[s])
        return vals[bs], regs[bs]

    def _gp_run_pack(self, names, Xtr, ytr, Xva, yva, reg_map, quick):
        Mtr = self._gp_pack_matrix(Xtr, names, reg_map)
        Mva = self._gp_pack_matrix(Xva, names, reg_map)
        scx = StandardScaler().fit(Mtr)
        scy = StandardScaler().fit(ytr.reshape(-1, 1))
        Ztr, Zva = scx.transform(Mtr), scx.transform(Mva)
        ztr = scy.transform(ytr.reshape(-1, 1)).flatten()
        zva = scy.transform(yva.reshape(-1, 1)).flatten()
        val, reg = self._gp_parallel(Ztr, ztr, Zva, zva, quick)
        return dict(reg=reg, scx=scx, scy=scy, names=list(names),
                    reg_map=reg_map, val=float(val))

    def _gp_solve(self, Xtr, ytr, Xva, yva, quick=False, wide=False):
        """阶梯 C -> B -> D；wide=True 时一次用更全积木（回流用）。"""
        reg_map = FC.reg_map_of(Xtr.shape[1])
        if wide:
            c_top = C_TOPK * 3
            names_c = self._gp_select_names(Xtr, ytr, ('d', 'p', 'r'), c_top)
            names_d = self._gp_select_names(Xtr, ytr, ('ppd', 'dpx'), D_TOPK * 2)
            names = names_c + names_d
            payload = self._gp_run_pack(
                names, Xtr, ytr, Xva, yva, reg_map, quick)
            payload['pack'] = 'wide'
            return payload
        packs = [
            ('C_pairs', self._gp_select_names(
                Xtr, ytr, ('d', 'p', 'r'), C_TOPK)),
            ('B_bare', []),
            ('D_deep', self._gp_select_names(
                Xtr, ytr, ('ppd', 'dpx'), D_TOPK)),
        ]
        best = None
        for pack_name, names in packs:
            payload = self._gp_run_pack(
                names, Xtr, ytr, Xva, yva, reg_map, quick)
            payload['pack'] = pack_name
            if best is None or payload['val'] > best['val']:
                best = payload
            if pack_name == 'C_pairs' and payload['val'] >= C_TH:
                break
            if pack_name == 'B_bare' and payload['val'] >= B_TH:
                break
        return best

    def _gp_predict(self, payload, X):
        M = self._gp_pack_matrix(X, payload['names'], payload['reg_map'])
        z = payload['reg'].predict(payload['scx'].transform(M))
        return payload['scy'].inverse_transform(
            np.asarray(z).reshape(-1, 1)).flatten()

    # ============================================================
    # 件② 激进路径（连续剥 conf>0.8，最终 r2 验证）
    # ============================================================
    def _try_complete(self, X, y, active, X1, cex1, deps1, col_expr, peels):
        keep = [j for j in range(X1.shape[1]) if deps1[j] <= active]
        r_base = -1.0
        if keep:
            s, c, r = self._additive_auto(
                X1, y, keep, self.seed + 300 + len(peels), exact=True)
            r_base = r
            if r >= SOLVE_R2:
                e, _ = self._assemble(s, c, cex1, col_expr, peels)
                return e, peels
        comp = self._peel_compound(
            X, y, active, X1, deps1, col_expr, r_base, len(peels), 0)
        if comp is not None:
            yt, factor = comp
            keep = [j for j in range(X1.shape[1]) if deps1[j] <= active]
            if keep:
                s, c, r2 = self._additive_auto(
                    X1, yt, keep, self.seed + 350 + len(peels), exact=True)
                if r2 >= SOLVE_R2:
                    e, _ = self._assemble(
                        s, c, cex1, col_expr, peels + [factor])
                    return e, peels + [factor]
        return None

    def _aggressive_path_legacy(self, X, y, col_expr):
        X1, cex1, deps1 = self._build_lean(X, col_expr)
        active = frozenset(range(X.shape[1]))
        yc = y.copy()
        peels = []
        for _ in range(MAX_PEEL):
            act_list = sorted(active)
            if not act_list:
                break
            st = PB.State(X[:, act_list], yc,
                          [('var', i) for i in range(len(act_list))],
                          ["x%d" % i for i in act_list], seed=self.seed)
            mi = PB.aggregate_observations(PB.scan_all(st, probes=[PB.MultProbe]))
            best_c, best_lk, best_p = 0.8, None, None
            for lk, (c, det) in mi['mult'].items():
                p = det.get('power') if det else None
                if p is None or abs(p - round(p)) > 1e-9 or round(p) == 0:
                    continue
                if c > best_c:
                    best_c, best_lk, best_p = c, lk, int(round(p))
            if best_lk is None:
                break
            orig_k = act_list[best_lk]
            yt = yc / np.power(np.clip(X[:, orig_k], 1e-9, None), best_p)
            if not np.all(np.isfinite(yt)):
                break
            factor = col_expr[orig_k] if best_p == 1 else \
                ('pow', col_expr[orig_k], best_p)
            active = active - {orig_k}
            yc = yt
            peels = peels + [factor]
            self.log.append("激进剥 %s^%s" % (self.names0[orig_k], best_p))
            done = self._try_complete(
                X, yc, active, X1, cex1, deps1, col_expr, peels)
            if done is not None:
                return done
        return None

    # G3/G4 统一 peel（连续剥乘性，不卡每步 PEEL_MIN；稳健 abs 探针 + 复合剥）
    _aggressive_path = UP.unified_peel
    _build_const = UP.build_const

    # ============================================================
    # 件③ 多切分复验 + 残差 signal（回流依据）
    # ============================================================
    def _cand_predict(self, c, X):
        if c[0] in ('exact', 'aggr', 'geom', 'rational', 'mult', 'outer_sqrt', 'additive', 'algebra', 'gaussian', 'trig', 'sg'):
            return _ev_c(c[1], X)
        return self._gp_predict(c[1], X)

    def _cand_cut_r2(self, c, X, y, ncuts=REVIEW_CUTS):
        n = len(y)
        scores = []
        for sd in range(ncuts):
            pm = np.random.default_rng(700 + sd).permutation(n)
            te = pm[int(0.75 * n):]
            try:
                pred = self._cand_predict(c, X[te])
                scores.append(float(SC.r2(y[te], pred)))
            except Exception:
                scores.append(-1.0)
        return scores

    def _resid_signal(self, c, X, y):
        pred = self._cand_predict(c, X)
        resid = y - pred
        if np.var(resid) < 1e-9:
            return 0.0
        Xr, cexr, depsr = self._build_lean(
            X, [('var', i) for i in range(X.shape[1])])
        try:
            _, _, rr = self._additive(
                Xr, resid, list(range(Xr.shape[1])),
                self.seed + 777, exact=False)
        except Exception:
            rr = 0.0
        return float(rr)

    @staticmethod
    def _mean_cut(cu):
        v = [x for x in cu if x == x]
        return float(np.mean(v)) if v else -1.0

    # ============================================================
    # 选择：精确解达标优先（符号真解），否则按多切分泛化选
    # ============================================================
    def _choose(self, scored):
        # 严格门：只有全量 r2 严格=1（无数值近似）的确定性解才允许"按最简"短路，
        # 防止 0.9999x 的数值近似伪装成精确解（III_19_51 事故：mult 线性组合
        # 近似 1/x3^2，r2=0.999994 挤掉 GP 真解）。
        exact = [(c, f, cu) for c, f, cu in scored
                 if c[0] in ('exact', 'aggr', 'geom', 'rational', 'mult', 'outer_sqrt', 'additive', 'algebra', 'gaussian', 'trig', 'sg')]
        good = [(c, f, cu) for c, f, cu in exact if f >= RATIONAL_R2]
        if good:
            good.sort(key=lambda t: (_nodes(t[0][1]), -t[1]))
            return good[0]
        # 无严格精确解：所有候选（含 GP）按 全量 r2 → 多切分 → 节点数 排序，
        # GP 真解（全量 r2=1）会胜过 0.9999x 的数值近似。
        if scored:
            order = sorted(scored,
                           key=lambda t: (-t[1], -self._mean_cut(t[2]),
                                          _nodes(t[0][1])))
            return order[0]
        return None, float('nan'), []

    # ============================================================
    # 对外入口（补全主流程）
    # ============================================================
    def fit(self, X=None, y=None, quick=False, max_time=None, **kw):
        X, y = self.X0, self.y0
        # Official max_time: global budget for the problem; when exceeded,
        # no new costly paths are entered and the current best candidate is
        # finalized.
        TSOL.set_global_budget(max_time)
        col_expr = [('var', i) for i in range(self.d0)]
        n = len(y)

        # J1 几何路径优先（快、确定性，约 2.5s）：模长/距离积木 + 剥 free。
        # 找到全量 r2≥SOLVE_R2 真解即短路，跳过昂贵的 _solve/激进/GP/复验。
        geom = GB.find_geometry(X, y, col_expr)
        if geom is not None and geom[0] >= SOLVE_R2:
            gcand = ('geom', geom[1], [])
            gfull = float(SC.r2(y, self._cand_predict(gcand, X)))
            gcuts = self._cand_cut_r2(gcand, X, y)
            self.family = 'geom'
            self.states = None
            self.log.append("几何路径命中 r2=%.5f 多切分=%.4f（短路昂贵路径）"
                            % (gfull, self._mean_cut(gcuts)))
            self._finalize(gcand, gfull, gcuts)
            return self

        # J2 分式复合反演（昂贵枚举，5740 分母候选）已后移到所有快速专用路径
        # 之后（见高斯族后）：非分式题被它全量无效扫描拖慢 100-300s，而快速路径
        # 0.2s 内可解；先跑快速路径，全部失败才用分式枚举兜底。

        # 纯乘性单项式路径（确定性）：log 空间多元最小二乘一次性求
        # 幂次向量（替代单变量贪心剥——多变量耦合时 conf 不稳，III_19_51 事故），
        # 模块内严格后验 r2；再全量复核 + 多切分稳定性，达标即短路昂贵路径。
        msol = MLS.solve_mult_log(X, y, col_expr)
        if msol is not None:
            mcand = ('mult', msol[1], [])
            mfull = float(SC.r2(y, self._cand_predict(mcand, X)))
            mcuts = self._cand_cut_r2(mcand, X, y)
            if mfull >= RATIONAL_R2 and self._mean_cut(mcuts) >= SOLVE_R2:
                self.family = 'mult'
                self.states = None
                self.log.append(
                    "纯乘性路径命中 r2=%.6f 多切分=%.5f（短路昂贵路径）"
                    % (mfull, self._mean_cut(mcuts)))
                self._finalize(mcand, mfull, mcuts)
                return self

        # J3 外层根号反演（确定性，纯乘之后）：y 同号 → z=y²，对每列对构造
        # 平方比值分母 g=1-(x_a/x_b)²（洛伦兹因子，现有 build_denom_factors 缺
        # u² 积木），w=z*g 用 MLS 解单项式分子 N，y=sgn*sqrt(N/g)，原始 y
        # 后验严格门 + 多切分稳定性，达标即短路昂贵路径。加性分子留 J6。
        osol = OSQ.find_outer_sqrt(X, y, col_expr)
        if osol is not None:
            ocand = ('outer_sqrt', osol[1], [])
            ofull = float(SC.r2(y, self._cand_predict(ocand, X)))
            ocuts = self._cand_cut_r2(ocand, X, y)
            if ofull >= RATIONAL_R2 and self._mean_cut(ocuts) >= SOLVE_R2:
                self.family = 'outer:sqrt'
                self.states = None
                self.log.append(
                    "外层根号反演命中 r2=%.6f 多切分=%.5f（短路昂贵路径）"
                    % (ofull, self._mean_cut(ocuts)))
                self._finalize(ocand, ofull, ocuts)
                return self

        # J6 B 加性递归分解（外层根号框架、分子为加性）：y=sgn*A/sqrt(g)，
        # 分子 A 在 q=sqrt(w) 上做加性列空间（常数/原始/pair 积/三元积商/两积
        # 除变量平方/双向比），不展开平方、不盲目搜索。d<=5（ADDITIVE_MAX_D）。
        # 达标即短路昂贵路径。
        asol = ADDB.find_additive_outer_sqrt(X, y, col_expr)
        if asol is not None:
            acand = ('additive', asol[1], [])
            afull = float(SC.r2(y, self._cand_predict(acand, X)))
            acuts = self._cand_cut_r2(acand, X, y)
            if afull >= RATIONAL_R2 and self._mean_cut(acuts) >= SOLVE_R2:
                self.family = 'outer:sqrt'
                self.states = None
                self.log.append(
                    "加性递归分解命中 r2=%.6f 多切分=%.5f（短路昂贵路径）"
                    % (afull, self._mean_cut(acuts)))
                self._finalize(acand, afull, acuts)
                return self

        # 根号内直接加性（不剥 g）：y=sqrt(加性积木)，积木含平方/平方、1/平方。
        isol = ADDB.find_sqrt_inner_additive(X, y, col_expr)
        if isol is not None:
            icand = ('additive', isol[1], [])
            ifull = float(SC.r2(y, self._cand_predict(icand, X)))
            icuts = self._cand_cut_r2(icand, X, y)
            if ifull >= RATIONAL_R2 and self._mean_cut(icuts) >= SOLVE_R2:
                self.family = 'outer:sqrt'
                self.states = None
                self.log.append(
                    "根号内加性命中 r2=%.6f 多切分=%.5f（短路昂贵路径）"
                    % (ifull, self._mean_cut(icuts)))
                self._finalize(icand, ifull, icuts)
                return self

        # 两列和（总质量/总量）后 MLS：覆盖 2π d^1.5/sqrt(G(m1+m2))。
        ssol = ADDB.find_sum_column_mls(X, y, col_expr)
        if ssol is not None:
            scand = ('additive', ssol[1], [])
            sfull = float(SC.r2(y, self._cand_predict(scand, X)))
            scuts = self._cand_cut_r2(scand, X, y)
            if sfull >= RATIONAL_R2 and self._mean_cut(scuts) >= SOLVE_R2:
                self.family = 'outer:sqrt'
                self.states = None
                self.log.append(
                    "和列MLS命中 r2=%.6f 多切分=%.5f（短路昂贵路径）"
                    % (sfull, self._mean_cut(scuts)))
                self._finalize(scand, sfull, scuts)
                return self

        # G1 共有乘性因子 + 两项加性（无根号）：y=C·(a1·t1+a2·t2)，
        # log 空间求共有因子 C，归一化后首项遍历 + 残差乘性，联合最小二乘；
        # 全量后验严格门 + 多切分，达标即短路。
        csol = CAB.find_common_additive(X, y, col_expr)
        if csol is not None:
            ccand = ('additive', csol[1], [])
            cfull = float(SC.r2(y, self._cand_predict(ccand, X)))
            ccuts = self._cand_cut_r2(ccand, X, y)
            if cfull >= RATIONAL_R2 and self._mean_cut(ccuts) >= SOLVE_R2:
                self.family = 'additive'
                self.states = None
                self.log.append(
                    "共有加性命中 r2=%.6f 多切分=%.5f（短路昂贵路径）"
                    % (cfull, self._mean_cut(ccuts)))
                self._finalize(ccand, cfull, ccuts)
                return self

        # J4 A 代数复合反演（玻尔兹曼 / 双曲外层）：y=Cn*Nstruct*core(u)，
        # core∈{exp(u)-1, 1/(exp(u)-1), 1/(2cosh u), tanh(u)}，Nstruct 为列
        # 简单乘性组合（常数/单列/pair 积/双向比/倒数），u 为单项式分数。
        # 反演后取可靠点（剔除饱和/奇点）定结构、MLS 求幂、全量严格后验。
        # d<=5（ALG_MAX_D）。达标即短路昂贵路径。
        alg = ALG.find_algebra(X, y, col_expr)
        if alg is not None:
            lcand = ('algebra', alg[1], [])
            lfull = float(SC.r2(y, self._cand_predict(lcand, X)))
            lcuts = self._cand_cut_r2(lcand, X, y)
            if lfull >= RATIONAL_R2 and self._mean_cut(lcuts) >= SOLVE_R2:
                self.family = 'algebra'
                self.states = None
                self.log.append(
                    "代数复合反演命中 r2=%.6f 多切分=%.5f（短路昂贵路径）"
                    % (lfull, self._mean_cut(lcuts)))
                self._finalize(lcand, lfull, lcuts)
                return self

        # G2 纯 exp 外层：y=x_j·exp(sgn·mono)，遍历变量试做乘性因子，
        # 归一化后 log 由其余变量乘性干净解释；全量严格门 + 多切分，达标即短路。
        esol = CAB.find_common_exp(X, y, col_expr)
        if esol is not None:
            ecand = ('additive', esol[1], [])
            efull = float(SC.r2(y, self._cand_predict(ecand, X)))
            ecuts = self._cand_cut_r2(ecand, X, y)
            if efull >= RATIONAL_R2 and self._mean_cut(ecuts) >= SOLVE_R2:
                self.family = 'algebra'
                self.states = None
                self.log.append(
                    "纯exp外层命中 r2=%.6f 多切分=%.5f（短路昂贵路径）"
                    % (efull, self._mean_cut(ecuts)))
                self._finalize(ecand, efull, ecuts)
                return self

        # J5 C 三角几何复合（反三角外层 + 余弦定理）：
        # ①反三角 y=arcsin/arccos/arctan(u)，扩展列(原始+sin/cos)上 MLS；
        # ②余弦定理 y=sqrt(a²+b²-2ab cos)，枚举角度项+边列对加性、系数 [1,1,-2]。
        # d<=5（TRIG_MAX_D）。达标即短路昂贵路径。
        if self.d0 <= TRIG.TRIG_MAX_D:
            tg = (TRIG.find_inverse_trig(X, y, col_expr)
                  or TRIG.find_inverse_mobius(X, y, col_expr)
                  or TRIG.find_cosine_law(X, y, col_expr)
                  or TRIG.find_cosine_law_reciprocal(X, y, col_expr)
                  or TRIG.find_interference_additive(X, y, col_expr)
                  or TRIG.find_wave_interference(X, y, col_expr))
            if tg is not None:
                lcand = ('trig', tg[1], [])
                lfull = float(SC.r2(y, self._cand_predict(lcand, X)))
                lcuts = self._cand_cut_r2(lcand, X, y)
                if lfull >= RATIONAL_R2 and self._mean_cut(lcuts) >= SOLVE_R2:
                    self.family = 'trig'
                    self.states = None
                    self.log.append(
                        "三角几何复合命中 r2=%.6f 多切分=%.5f（短路昂贵路径）"
                        % (lfull, self._mean_cut(lcuts)))
                    self._finalize(lcand, lfull, lcuts)
                    return self

        # 高斯族：y=exp(-u²/2)/(sqrt(2pi) s)，宽度单列、位置 单列/两列差。
        gz = GAUS.find_gaussian(X, y, col_expr)
        if gz is not None:
            lcand = ('gaussian', gz[1], [])
            lfull = float(SC.r2(y, self._cand_predict(lcand, X)))
            lcuts = self._cand_cut_r2(lcand, X, y)
            if lfull >= RATIONAL_R2 and self._mean_cut(lcuts) >= SOLVE_R2:
                self.family = 'gaussian'
                self.states = None
                self.log.append(
                    "高斯族命中 r2=%.6f 多切分=%.5f（短路昂贵路径）"
                    % (lfull, self._mean_cut(lcuts)))
                self._finalize(lcand, lfull, lcuts)
                return self

        # Unified assume-transform-verify solver (combined solve): first a
        # full-column greedy, then transform_solve peeling (multiplicative /
        # angle factors / denominators / outer sqrt / composite powers).
        # Cost 3-12s, far cheaper than rational enumeration (120s), and it
        # solves several problems, so it is placed first; a result with r2=1,
        # passing the symbolic gate and stable across splits short-circuits.
        try:
            usol = TSOL.combined_solve(X, y, self.names0)
        except Exception as ex:
            usol = None
            self.log.append("统一求解器异常: %r" % ex)
        if usol is not None:
            unode = usol[1]
            ufull = float(SC.r2(y, _ev_c(unode, X)))
            if ufull >= RATIONAL_R2 and SG.symbol_gate(unode):
                ucand = ('sg', unode, None)
                ucuts = self._cand_cut_r2(ucand, X, y)
                if self._mean_cut(ucuts) >= SOLVE_R2:
                    self.family = 'unified'
                    self.states = None
                    self.log.append(
                        "统一求解器短路 r2=%.6f 多切分=%.5f（跳过分式枚举/solve/GP）"
                        % (ufull, self._mean_cut(ucuts)))
                    self._finalize(ucand, ufull, ucuts)
                    return self

        # Structure-guided solve (peel + shift family + multiplicative
        # large/small term family): placed before the two costly rational
        # enumerations. Full r2=1 with the symbolic gate passing (simple
        # terms/coefficients) is the real solution and short-circuits
        # rational/solve/aggr/GP; if the gate fails, rational enumeration
        # continues as the fallback.
        sg = None
        try:
            sg = SG.structure_guided_solve(X, y, self.names0)
        except Exception as ex:
            self.log.append("结构引导异常: %r" % ex)
        if sg is not None:
            sg_node = sg[1]
            sg_full = float(SC.r2(y, _ev_c(sg_node, X)))
            if sg_full >= RATIONAL_R2 and SG.symbol_gate(sg_node):
                sg_cand = ('sg', sg_node, None)
                sg_cuts = self._cand_cut_r2(sg_cand, X, y)
                self.log.append(
                    "结构引导短路 r2=%.6f 符号门通过（跳过rational/solve/aggr/GP）"
                    % sg_full)
                self._finalize(sg_cand, sg_full, sg_cuts)
                return self

        # J2 分式复合反演（昂贵枚举，5740 分母候选 × 2 p）：统一求解器也失败后才
        # 兜底（统一求解 3-12s 已覆盖大部分分式题），避免非分式题被它全量无效扫描
        # 拖慢 100-300s。阶段1 打分仅排序，真值靠阶段2 后验 + 符号等价核验。
        if (RATIONAL_ENABLED and self.d0 <= RATIONAL_MAX_D
                and not TSOL._deadline_hit()):
            rat = RB.find_rational(self, X, y, col_expr)
            if rat is not None and rat[0] >= SOLVE_R2:
                rcand = ('rational', rat[1], [])
                rfull = float(SC.r2(y, self._cand_predict(rcand, X)))
                rcuts = self._cand_cut_r2(rcand, X, y)
                # 主门：全量 r2 严格=1（RATIONAL_R2）；多切分 mean≥SOLVE_R2
                # 作稳定性（防全量巧合）。结构错假解 r2 卡在 .999x 被主门拦下。
                if rfull >= RATIONAL_R2 and self._mean_cut(rcuts) >= SOLVE_R2:
                    self.family = 'rational'
                    self.states = None
                    self.log.append(
                        "分式路径命中 r2=%.6f 多切分=%.5f（短路昂贵路径）"
                        % (rfull, self._mean_cut(rcuts)))
                    self._finalize(rcand, rfull, rcuts)
                    return self

        # J5b 分式外框 + 分子 sqrt（sqrt 内加性）：y = sqrt(P)/g。find_rational 剥
        # 分母后分子带 sqrt、不在其线性列空间；本函数先过 g=1 假分式门（无分母即
        # 有精确稀疏解 → 纯 sqrt 题，返回 None），再枚举分母剥 g 后在 z² 上做严格
        # 加性（精确 r2≥1-1e-9 + 稀疏 nterms≤8 + 分数系数重建 三重门），覆盖相对
        # 论多普勒角频率。**放在所有确定性快路径之后**：能被快路径短路的题不跑此
        # 枚举（避免非分式题被拖慢），仅快路径全失败的残题才试。d<=5。达标即短路。
        if (RATIONAL_ENABLED and self.d0 <= RATIONAL_MAX_D
                and not TSOL._deadline_hit()):
            rsq = RB.find_rational_sqrt(X, y, col_expr)
            if rsq is not None and rsq[0] >= SOLVE_R2:
                rscand = ('rational', rsq[1], [])
                rsfull = float(SC.r2(y, self._cand_predict(rscand, X)))
                rscuts = self._cand_cut_r2(rscand, X, y)
                if rsfull >= RATIONAL_R2 and self._mean_cut(rscuts) >= SOLVE_R2:
                    self.family = 'rational'
                    self.states = None
                    self.log.append(
                        "分式sqrt路径命中 r2=%.6f 多切分=%.5f（短路昂贵路径）"
                        % (rsfull, self._mean_cut(rscuts)))
                    self._finalize(rscand, rsfull, rscuts)
                    return self

        # Global budget exhausted (official max_time): no more solve/aggr/GP;
        # finalize capability_limit.
        if TSOL._deadline_hit():
            self.log.append("Global budget exhausted (max_time), capability_limit")
            self._finalize(None, float('nan'), None)
            return self

        # Geometry/fraction not passed -> structure recognition + existing
        # costly paths (solve/aggr/GP/re-validation)
        mi, family, states = self._recognize(X, y)
        self.family = family
        self.states = states
        pm = np.random.default_rng(self.seed).permutation(n)
        cut = int(0.75 * n)
        tr, va = pm[:cut], pm[cut:]

        # 精确解（V7 现有：直接解 + 外层）
        cands = []
        exact_res = self._solve(X.copy(), y.copy(), col_expr, 0)
        if exact_res is not None:
            e_exact = exact_res[0]
            r_exact = float(SC.r2(y, _ev_c(e_exact, X)))
            ecand = ('exact', e_exact, exact_res[1])
            # r2=1 精确解 + 多切分稳定 → 真解，短路 aggr/GP（避免重复穷举）
            if r_exact >= RATIONAL_R2:
                ecuts = self._cand_cut_r2(ecand, X, y)
                if self._mean_cut(ecuts) >= SOLVE_R2:
                    self.log.append(
                        "精确解短路 r2=%.6f 多切分=%.5f（跳过aggr/GP）"
                        % (r_exact, self._mean_cut(ecuts)))
                    self._finalize(ecand, r_exact, ecuts)
                    return self
            cands.append(('exact', e_exact, exact_res[1]))
        # 激进路径
        aggr = self._aggressive_path(X, y, col_expr)
        if aggr is not None:
            cands.append(('aggr', aggr[0], aggr[1]))
        # 结构引导候选（符号门不通过的仍交 _choose 比较，但 _finalize 不判 exact）
        if sg is not None:
            cands.append(('sg', sg[1], None))
        # GP 兜底
        gp = self._gp_solve(X[tr], y[tr], X[va], y[va], quick=quick)
        if gp is not None:
            cands.append(('gp', gp, None))

        # 评分：全量 r2 + 多切分
        scored = []
        for c in cands:
            full = float(SC.r2(y, self._cand_predict(c, X)))
            cuts = self._cand_cut_r2(c, X, y)
            scored.append((c, full, cuts))
            self.log.append("候选%s 全量r2=%.4f 多切分均值=%.4f" % (
                c[0], full, self._mean_cut(cuts)))

        choice, full_r2, cut_r2 = self._choose(scored)

        # 复验回流：残差 signal 或临界不稳 → 扩列（wide）再 GP
        for rnd in range(MAX_REFLOW):
            if choice is None:
                break
            sig = self._resid_signal(choice, X, y)
            unstable = self._mean_cut(cut_r2) < REVIEW_THRESH or \
                min([x for x in cut_r2 if x == x] or [-1]) < REVIEW_THRESH
            if not (sig >= 0.5 or unstable):
                break
            gp_wide = self._gp_solve(
                X[tr], y[tr], X[va], y[va], quick=quick, wide=True)
            if gp_wide is None:
                break
            wfull = float(SC.r2(y, self._gp_predict(gp_wide, X)))
            wcuts = self._cand_cut_r2(('gp', gp_wide, None), X, y)
            self.log.append("回流%d wide 全量=%.4f 多切分=%.4f 残差signal=%.3f"
                            % (rnd + 1, wfull, self._mean_cut(wcuts), sig))
            if self._mean_cut(wcuts) > self._mean_cut(cut_r2) or \
                    (wfull > full_r2 and self._mean_cut(wcuts) >=
                     self._mean_cut(cut_r2) - 0.01):
                choice, full_r2, cut_r2 = ('gp', gp_wide, None), wfull, wcuts
            else:
                break

        # Finalize
        self._finalize(choice, full_r2, cut_r2)
        return self

    def _finalize(self, choice, full_r2, cut_r2):
        self._cut_r2 = cut_r2
        if choice is None:
            self._cut_mean = float('nan')
            self.state = 'capability_limit'
            self._r2v = float('nan')
            self._expr = None
            self._kind = None
            return
        self._cut_mean = self._mean_cut(cut_r2)
        if choice[0] in ('exact', 'aggr', 'geom', 'rational', 'mult', 'outer_sqrt', 'additive', 'algebra', 'gaussian', 'trig', 'sg'):
            # exact 放行门：确定性来源不等于符号真解。数值 r2 必须严格=1
            # （全量或多切分均值达 RATIONAL_R2），否则只是高精度数值近似，
            # 标 kind=approx，不冒充 exact（符号真解仍由 fit 后符号等价核验复核）。
            strict = (full_r2 >= RATIONAL_R2
                      or self._cut_mean >= RATIONAL_R2)
            # sg 额外要求符号门通过（项数/系数简单）；高精度展开（如 test_17
            # 10 项）即使 r2=1 也不判 exact，只算 approx。
            if choice[0] == 'sg' and not SG.symbol_gate(choice[1]):
                strict = False
            self._kind = 'exact' if strict else 'approx'
            self._expr = choice[1]
            self.peels = choice[2]
        else:
            self._kind = 'gp'
            self._expr = None
            self._gp_payload = choice[1]
        self._r2v = full_r2
        # 最终表达式过符号 cancel（消冗余互逆因子，如 A×1/A），用于 predict；
        # 纯代数恒等化简，失败则 None、predict 回退 _ev_c。
        self._clean_lambda = None
        if self._expr is not None and self._kind == 'exact':
            self._clean_lambda = self._build_clean_lambda()
        # solved 判据：全量高 或 多切分泛化达标
        ok = full_r2 >= 0.99 or self._cut_mean >= REVIEW_THRESH
        self.state = 'solved' if ok else 'capability_limit'
        self.log.append("Finalize kind=%s full_r2=%.4f multi-split mean=%.4f state=%s"
                        % (self._kind, full_r2, self._cut_mean, self.state))

    def _build_clean_lambda(self):
        """tuple 最终表达式 -> 符号 cancel -> lambdify（数值用）。

        纯代数恒等化简（消冗余互逆因子、分式归一）；仅用训练数据自检，
        不碰真值。化简失败 / 数值不一致 / 出现非有限值，返回 None。
        """
        import sympy as sp
        import symbol_simplify as SS
        try:
            cleaned = SS.clean_sympy(self._expr, list(self.names0), timeout=30)
            if cleaned is None:
                return None
            syms = [sp.Symbol(n) for n in self.names0]
            fn = sp.lambdify(syms, cleaned, modules='numpy')
            X = self.X0
            args = [X[:, i] for i in range(X.shape[1])]
            pred = np.asarray(fn(*args), dtype=float)
            if pred.shape != (len(X),) or not np.all(np.isfinite(pred)):
                return None
            denom = np.sum((self.y0 - np.mean(self.y0)) ** 2)
            r2 = 1 - np.sum((self.y0 - pred) ** 2) / denom
            if r2 < 1 - 1e-6:
                return None
            return fn
        except Exception:
            return None

    def predict(self, X):
        X = np.asarray(X, float)
        if self._kind == 'gp':
            return self._gp_predict(self._gp_payload, X)
        Xc = X
        if self._protect_cols:
            Xc = X.copy()
            for k in self._protect_cols:
                lo, hi = self._bounds[k]
                Xc[:, k] = np.clip(Xc[:, k], lo * 0.99, hi * 1.01)
        # 优先用符号 cancel 后的干净表达式（消冗余互逆因子、数值稳定）
        if getattr(self, '_clean_lambda', None) is not None:
            try:
                args = [Xc[:, i] for i in range(Xc.shape[1])]
                out = np.asarray(self._clean_lambda(*args), dtype=float)
                if out.shape == (len(Xc),) and np.all(np.isfinite(out)):
                    return out
            except Exception:
                pass
        return _ev_c(self._expr, Xc)

    def get_kind(self):
        return self._kind

    def get_cut_r2(self):
        return getattr(self, '_cut_r2', None)

    def get_gp_string(self):
        if self._kind != 'gp':
            return None
        reg = self._gp_payload['reg']
        try:
            return reg.get_model_string(reg.model_)
        except Exception:
            return None


def fit_engine_full(X, y, names, seed=0, quick=False, max_time=None):
    return SHRv7Full(X, y, names, seed=seed).fit(quick=quick, max_time=max_time)
