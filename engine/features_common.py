# -*- coding: utf-8 -*-
"""合规通用特征注册表（name 唯一、可在任意 X 上重算、不读真值）。

规则（对所有题统一，仅依赖列索引，不含任何针对单题的分支）：
  两两：d{p}{q} 差、p{p}{q} 积、r{p}{q} 比（双向）
  三元（每个三组合）：
    tpp{abc}              x_a*x_b*x_c
    ppd{ab}_{c}           (x_a*x_b)/x_c   （分子两列、分母一列，三轮换唯一名）
    dpx{ab}_{c}           (x_a-x_b)*x_c   （差×第三变量，三轮换）
  立方：c{i}              x_i**3
"""
from itertools import combinations
import numpy as np

_SAFE_MAX = 1e6


def _safe(col):
    col = np.asarray(col, float)
    if np.all(np.isfinite(col)) and np.max(np.abs(col)) < _SAFE_MAX:
        return col
    return None


def _ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(np.abs(b) > 1e-12, a / np.where(np.abs(b) > 1e-12, b, 1.0), np.nan)
    return r


def build_registry(d):
    """返回 [(name, fn(X)->col)]，fn 仅依赖列索引。"""
    reg = []

    def add(name, fn):
        reg.append((name, fn))

    for i in range(d):
        add("c%d" % (i + 1), lambda X, i=i: X[:, i] ** 3)
    for a in range(d):
        for b in range(a + 1, d):
            add("d%d%d" % (a + 1, b + 1), lambda X, a=a, b=b: X[:, a] - X[:, b])
            add("p%d%d" % (a + 1, b + 1), lambda X, a=a, b=b: X[:, a] * X[:, b])
            add("r%d%d" % (a + 1, b + 1), lambda X, a=a, b=b: _ratio(X[:, a], X[:, b]))
            add("r%d%d" % (b + 1, a + 1), lambda X, a=a, b=b: _ratio(X[:, b], X[:, a]))
    for a, b, c in combinations(range(d), 3):
        add("tpp%d%d%d" % (a + 1, b + 1, c + 1),
            lambda X, a=a, b=b, c=c: X[:, a] * X[:, b] * X[:, c])
        # (x*x)/x 三轮换，name 唯一
        add("ppd%d%d_%d" % (a + 1, b + 1, c + 1),
            lambda X, a=a, b=b, c=c: X[:, a] * X[:, b] / X[:, c])
        add("ppd%d%d_%d" % (a + 1, c + 1, b + 1),
            lambda X, a=a, c=c, b=b: X[:, a] * X[:, c] / X[:, b])
        add("ppd%d%d_%d" % (b + 1, c + 1, a + 1),
            lambda X, b=b, c=c, a=a: X[:, b] * X[:, c] / X[:, a])
        # 差×第三变量 三轮换
        add("dpx%d%d_%d" % (a + 1, b + 1, c + 1),
            lambda X, a=a, b=b, c=c: (X[:, a] - X[:, b]) * X[:, c])
        add("dpx%d%d_%d" % (a + 1, c + 1, b + 1),
            lambda X, a=a, c=c, b=b: (X[:, a] - X[:, c]) * X[:, b])
        add("dpx%d%d_%d" % (b + 1, c + 1, a + 1),
            lambda X, b=b, c=c, a=a: (X[:, b] - X[:, c]) * X[:, a])
    return reg


def build_all(X):
    """在给定 X 上生成全部有效候选，返回 {name: col}（保持注册顺序）。"""
    out = {}
    for name, fn in build_registry(X.shape[1]):
        col = _safe(fn(X))
        if col is not None:
            out[name] = col
    return out


def eval_subset(names, X, reg_map):
    """按 names 顺序在 X 上重算，返回列矩阵；用 reg_map 里的 fn。"""
    cols = []
    for name in names:
        col = _safe(reg_map[name](X))
        if col is None:
            col = np.zeros(X.shape[0])
        cols.append(col)
    return np.column_stack(cols)


def reg_map_of(d):
    return {name: fn for name, fn in build_registry(d)}
