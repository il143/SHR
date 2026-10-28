# -*- coding: utf-8 -*-
"""SHR J1 几何确定性复合库（需要 y，独立于纯 X 的 features_common）。

几何积木（只依赖列索引，对所有题同一规则，不读题名/真值）：
  模长（原点）n2 / n3：
    sqrt(x_a²+x_b²)、sqrt(x_a²+x_b²+x_c²)
  两点距离 2 维（配对枚举）：
    sqrt((x_a-x_c)²+(x_b-x_d)²)
  两点距离² 3 维（配对枚举，无外层 sqrt）：
    (x_a-x_d)²+(x_b-x_e)²+(x_c-x_f)²

几何积木本身不含"系数/缩放"信息；系数 = 积木依赖之外的 free 变量。
free≤3 时穷举 free 幂次（-1=y/x、0=不剥、+1=y*x）×（y~g 或 y~1/g），
可靠；free>3 时 greedy 兜底。r2 对常数倍不变。

服务题目（已机制验证，见 eval_full/_j1_geom_probe.out）：
  III_10_19  mom*sqrt(Bx²+By²+Bz²)
  I_8_14     sqrt((x2-x1)²+(y2-y1)²)
  I_9_18     G*m1*m2/((x2-x1)²+(y2-y1)²+(z2-z1)²)
"""
from itertools import combinations, product
import numpy as np


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    mask = np.isfinite(p)
    if mask.sum() < 5:
        return -1.0
    y, p = y[mask], p[mask]
    ss = np.sum((y - y.mean()) ** 2)
    if ss < 1e-12:
        return 1.0 if np.var(p) < 1e-9 else -1.0
    return 1.0 - np.sum((y - p) ** 2) / ss


def _pairings(indices, k):
    """把 2k 个 indices 分两组（每组 k），以第一个为锚去重。"""
    indices = list(indices)
    first = indices[0]
    rest = indices[1:]
    for comp in combinations(rest, k - 1):
        P = tuple(sorted([first] + list(comp)))
        Q = tuple(sorted(i for i in indices if i not in P))
        yield P, Q


def _sq(x):
    return ('pow', x, 2)


def build_blocks(X, col_expr):
    """返回 [(g_vec, g_expr_rel_X0, dep)]。"""
    d = X.shape[1]
    out = []

    # 模长 2 维（原点）
    for a, b in combinations(range(d), 2):
        g = np.sqrt(X[:, a] ** 2 + X[:, b] ** 2)
        e = ('sqrt', ('add', _sq(col_expr[a]), _sq(col_expr[b])))
        out.append((g, e, frozenset([a, b])))
    # 模长 3 维（原点）
    for a, b, c in combinations(range(d), 3):
        g = np.sqrt(X[:, a] ** 2 + X[:, b] ** 2 + X[:, c] ** 2)
        inner = ('add', ('add', _sq(col_expr[a]), _sq(col_expr[b])),
                 _sq(col_expr[c]))
        out.append((g, ('sqrt', inner), frozenset([a, b, c])))
    # 两点距离 2 维
    for four in combinations(range(d), 4):
        for P, Q in _pairings(four, 2):
            g = np.sqrt(sum((X[:, p] - X[:, q]) ** 2
                            for p, q in zip(P, Q)))
            inner = ('add',
                     _sq(('sub', col_expr[P[0]], col_expr[Q[0]])),
                     _sq(('sub', col_expr[P[1]], col_expr[Q[1]])))
            out.append((g, ('sqrt', inner), frozenset(four)))
    # 两点距离² 3 维
    for six in combinations(range(d), 6):
        for P, Q in _pairings(six, 3):
            g = sum((X[:, p] - X[:, q]) ** 2 for p, q in zip(P, Q))
            inner = ('add',
                     ('add',
                      _sq(('sub', col_expr[P[0]], col_expr[Q[0]])),
                      _sq(('sub', col_expr[P[1]], col_expr[Q[1]]))),
                     _sq(('sub', col_expr[P[2]], col_expr[Q[2]])))
            out.append((g, inner, frozenset(six)))
    return out


def _peel_signs(y, X, free, signs):
    """按 signs {var: -1/+1} 剥，返回 y' 或 None。"""
    yt = y
    for a, s in signs.items():
        if s == -1:
            yt = yt / np.clip(X[:, a], 1e-9, None)
        else:
            yt = yt * X[:, a]
    if not np.all(np.isfinite(yt)):
        return None
    return yt


def _assemble(col_expr, g_expr, rel, signs):
    """系数因子 × 核心（rel=0 核心=g，rel=1 核心=1/g）。"""
    core = g_expr if rel == 0 else ('div', ('const', 1.0), g_expr)
    for a, s in signs.items():
        if s == -1:          # y = x_a * 核心
            core = ('mul', col_expr[a], core)
        else:                # y = 核心 / x_a
            core = ('div', core, col_expr[a])
    return core


def find_geometry(X, y, col_expr):
    """几何积木作为额外列，纯乘性 log 解统一求解。

    旧版 free≤3 穷举 ±1、free>3 greedy 剥离，无法处理 r^-5 这类非 ±1 幂次
    和 4 个以上 free（II_6_15a 因此失败）。改为把每个几何积木 g 当作一个
    额外列，交 mult_log_solver 一次性求全部幂次（含积木列），自动处理任意
    整数幂、奇次符号、数值系数；错误结构由 mult_log 的严格 r2 门拦住。
    返回 (r2, expr) 或 None。"""
    import mult_log_solver as MLS
    best = None
    for g, ge, dep in build_blocks(X, col_expr):
        Xext = np.column_stack([X, g])
        cext = list(col_expr) + [ge]
        r = MLS.solve_mult_log(Xext, y, cext)
        if r is not None and (best is None or r[0] > best[0]):
            best = (r[0], r[1])
    return best
