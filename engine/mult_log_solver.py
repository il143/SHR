# -*- coding: utf-8 -*-
"""纯乘性单项式确定性解（mult family 通用解）。

结构：y = C * ∏ x_j^k_j （k_j 为整数，奇次幂含符号）。

机理：
  1) log 空间多元最小二乘  log|y| = log|C| + Σ k_j log|x_j|，一次性求全幂次，
     避免单变量探针在多变量乘积耦合时 conf 被压低、贪心剥失败；
  2) 幂次系数就近取整，须干净（偏离整判 < COEF_TOL、|k|<=POWER_MAX）；
  3) 用幂次单项式（含奇次符号）重建 y，常数 C 带符号最小二乘，
     全量数值 r2 严格门（SOLVE_R2）才判真解——后验门会拦下
     arcsin(z)≈z 等"假乘性"，不依赖 log 拟合的高 r2 直接下结论。

探针（log 回归）是评估，真值靠全量后验；无逐题分支、无真值依赖。
"""
import numpy as np

_EPS = 1e-9
POWER_MAX = 6        # 幂次绝对值上限
COEF_TOL = 0.05      # 幂次系数偏离整判上限
SOLVE_R2 = 1.0 - 1e-7


def solve_mult_log(X, y, col_expr, kmax=POWER_MAX):
    """返回 (r2, expr_tuple, C) 或 None。"""
    ok = np.abs(y) > _EPS
    for j in range(X.shape[1]):
        ok &= np.abs(X[:, j]) > _EPS
    Xo, yo = X[ok], y[ok]
    if len(yo) < X.shape[1] + 2:
        return None
    A = np.column_stack([np.ones(len(yo)), np.log(np.abs(Xo))])
    coef = np.linalg.lstsq(A, np.log(np.abs(yo)), rcond=None)[0]
    k = coef[1:]
    ki = [int(round(v)) for v in k]
    if any(abs(v - r) > COEF_TOL or abs(r) > kmax for v, r in zip(k, ki)):
        return None

    S = np.ones(len(y))
    terms = []
    for j, kj in enumerate(ki):
        if kj == 0:
            continue
        ax = np.abs(X[:, j])
        if kj % 2 == 0:
            S = S * np.power(ax, kj)
        else:
            S = S * np.sign(X[:, j]) * np.power(ax, kj)
        if kj == 1:
            terms.append(col_expr[j])
        elif kj > 0:
            terms.append(('pow', col_expr[j], kj))
        elif kj == -1:
            terms.append(('div', ('const', 1.0), col_expr[j]))
        else:
            terms.append(('div', ('const', 1.0), ('pow', col_expr[j], -kj)))

    C = float(np.dot(S, y) / max(np.dot(S, S), _EPS))
    ss = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - np.sum((y - C * S) ** 2) / max(ss, 1e-12))
    if r2 < SOLVE_R2:
        return None
    # _ev_c 的 'mul' 是二元，左结合嵌套：mul(mul(mul(const,t1),t2),t3)...
    expr = ('const', C)
    for term in terms:
        expr = ('mul', expr, term)
    return r2, expr, C
