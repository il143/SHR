# -*- coding: utf-8 -*-
"""结构性检测工具（AI Feynman 式，独立模块，不碰现有求解流程）。

只在"现有流程解不出 exact"时兜底启用。纯数值、不读真值、不逐题写死。
统一机理：若 f 只通过某个二元组合 c(x_i,x_j) 依赖这两个变量，
则存在"保持 c 不变、改变 x_i,x_j"的变换 T_c，使 f 不变。

支持组合：
  diff   a-b     （平移对称：a+t,b+t）
  sum    a+b     （a+t,b-t）
  prod   a*b     （a(1+t),b/(1+t)）
  ratio  a/b     （a(1+t),b(1+t)）
  sqdiff a**2-b**2（sqrt(a**2+d),sqrt(b**2+d)）
  sqsum  a**2+b**2（旋转）
"""
import numpy as np

# 各组合的变换参数（中心采样留 30% 余量，参数保证在域内）
# 线性类（diff/sum）：参数相对 span，绝对量
# 乘性类（prod/ratio）：参数是相对比例
# 平方类：sqdiff 相对 span²，sqsum 是弧度
T_PARAMS = {
    "diff": (0.03, 0.07, 0.12),
    "sum": (0.03, 0.07, 0.12),
    "prod": (0.05, 0.10, 0.18),
    "ratio": (0.05, 0.10, 0.18),
    "sqdiff": (0.02, 0.05, 0.10),
    "sqsum": (0.05, 0.10, 0.18),
}

HARD_MED = 0.005
HARD_P90 = 0.02
SOFT_MED = 0.01
SOFT_P90 = 0.04


def _transform(comp, a, b, t, span):
    """返回保持组合不变的 (a',b')。a,b 为正（Feynman 数据 1~10）。"""
    if comp == "diff":
        d = t * span
        return a + d, b + d
    if comp == "sum":
        d = t * span
        return a + d, b - d
    if comp == "prod":
        f = 1.0 + t
        return a * f, b / f
    if comp == "ratio":
        f = 1.0 + t
        return a * f, b * f
    if comp == "sqdiff":
        dd = t * span ** 2
        return np.sqrt(a ** 2 + dd), np.sqrt(b ** 2 + dd)
    if comp == "sqsum":
        c, s = np.cos(t), np.sin(t)
        return a * c - b * s, a * s + b * c
    raise ValueError(comp)


def _combo_expr(comp, i, j):
    """返回组合的节点表达式。"""
    vi, vj = ("c", i), ("c", j)
    if comp == "diff":
        return ("sub", vi, vj)
    if comp == "sum":
        return ("add", vi, vj)
    if comp == "prod":
        return ("mul", vi, vj)
    if comp == "ratio":
        return ("div", vi, vj)
    if comp == "sqdiff":
        return ("sub", ("pow", vi, 2), ("pow", vj, 2))
    if comp == "sqsum":
        return ("add", ("pow", vi, 2), ("pow", vj, 2))
    raise ValueError(comp)


def _combo_value(comp, X, i, j):
    a, b = X[:, i], X[:, j]
    if comp == "diff":
        return a - b
    if comp == "sum":
        return a + b
    if comp == "prod":
        return a * b
    if comp == "ratio":
        return a / b
    if comp == "sqdiff":
        return a ** 2 - b ** 2
    if comp == "sqsum":
        return a ** 2 + b ** 2
    raise ValueError(comp)


def check_one(S, X, i, j, comp, n=200, seed=91, margin=0.30):
    """检测 f 是否只通过组合 comp 依赖 (i,j)。返回 (conf, med, p90)。"""
    lo, hi = X.min(0), X.max(0)
    span = hi - lo
    rng = np.random.default_rng(seed + 7 * i + 13 * j)
    P = np.empty((n, X.shape[1]))
    for k in range(X.shape[1]):
        m = lo[k] + margin * span[k]
        M = hi[k] - margin * span[k]
        if M <= m:
            m, M = lo[k], hi[k]
        P[:, k] = rng.uniform(m, M, n)
    f1 = S.query(P)
    worst_med, worst_p90 = 0.0, 0.0
    for t in T_PARAMS[comp]:
        P2 = P.copy()
        a2, b2 = _transform(comp, P[:, i], P[:, j], t, span[i])
        P2[:, i], P2[:, j] = a2, b2
        inside = np.all((P2 >= lo) & (P2 <= hi), axis=1)
        finite = np.all(np.isfinite(P2), axis=1)
        keep = inside & finite
        if keep.mean() < 0.5:
            continue
        f2 = S.query(P2)
        r = np.abs(f1 - f2) / (np.abs(f1) + np.abs(f2) + 1e-9)
        r = r[keep]
        med, p90 = float(np.median(r)), float(np.quantile(r, 0.9))
        worst_med = max(worst_med, med)
        worst_p90 = max(worst_p90, p90)
    if worst_med < HARD_MED and worst_p90 < HARD_P90:
        conf = 0.90 + 0.10 * (1 - worst_med / HARD_MED)
    elif worst_med < SOFT_MED and worst_p90 < SOFT_P90:
        conf = 0.55 + 0.35 * (1 - worst_med / SOFT_MED)
    else:
        conf = 0.25 * (1 - min(1.0, worst_med / 0.05))
    return float(min(1.0, max(0.0, conf))), worst_med, worst_p90


def detect(S, X, comps=None, hard_only=True, verbose=False):
    """扫描所有变量对 × 组合，返回检测到的结构列表。

    返回 [{'i','j','comp','conf','expr','value'}]，按置信度降序。
    """
    comps = comps or ("diff", "sum", "prod", "ratio", "sqdiff", "sqsum")
    d = X.shape[1]
    found = []
    for i in range(d):
        for j in range(i + 1, d):
            for comp in comps:
                conf, med, p90 = check_one(S, X, i, j, comp)
                if verbose:
                    print("(%d,%d) %-7s conf=%.3f med=%.5f p90=%.5f" %
                          (i, j, comp, conf, med, p90))
                if (hard_only and conf >= 0.9) or (not hard_only and conf >= 0.55):
                    found.append({
                        "i": i, "j": j, "comp": comp, "conf": conf,
                        "expr": _combo_expr(comp, i, j),
                        "value": _combo_value(comp, X, i, j),
                    })
    found.sort(key=lambda r: -r["conf"])
    return found
