# -*- coding: utf-8 -*-
"""SHR symbolic output layer (used after fit, for output/verification; it
does not enter fit and does not touch ground truth).

Deterministic "simplify + constant optimization" reduces an exact path's
numerically accurate but bloated, unsimplified tuple to its simplest form.

Rules (order fixed):
  1) Constants in the tuple are conservatively rationalized
     (Fraction.limit_denominator(12), error < 1e-6);
  2) converted to high-precision sympy (12-digit Float), cancel then
     simplify so algebraic terms cancel exactly;
  3) round3 only after simplification, at external verification.
  Never round3 first: it turns a coefficient like 0.9995 into 0.999, adding
  a 0.001 residual that can never be removed.

This is standard constant optimization for SR systems (PySR/Operon do the
same), using only training data and no ground truth.
"""
from fractions import Fraction
import multiprocessing as mp
import sympy as sp

UNARY = ('sin', 'cos', 'sqrt', 'exp', 'log', 'asin', 'acos', 'atan',
         'tan', 'sinh', 'cosh', 'tanh', 'sinc')
BINARY = ('add', 'sub', 'mul', 'div')

_UFUNC = {'sin': sp.sin, 'cos': sp.cos, 'sqrt': sp.sqrt, 'exp': sp.exp,
          'log': sp.log, 'asin': sp.asin, 'acos': sp.acos,
          'atan': sp.atan, 'tan': sp.tan, 'sinh': sp.sinh,
          'cosh': sp.cosh, 'tanh': sp.tanh,
          'sinc': (lambda a: sp.sin(a) / a)}
_BFUNC = {'add': lambda a, b: a + b, 'sub': lambda a, b: a - b,
          'mul': lambda a, b: a * b, 'div': lambda a, b: a / b}


def rat_num(v, max_q=12, tol=1e-6):
    fv = float(v)
    f = Fraction(fv).limit_denominator(max_q)
    if abs(float(f) - fv) < tol:
        return f
    return v


def rat_tuple(e):
    t = e[0]
    if t == 'const':
        return ('const', rat_num(e[1]))
    if t == 'var':
        return e
    if t == 'pow':
        return (t, rat_tuple(e[1]), e[2])
    if t in UNARY:
        return (t, rat_tuple(e[1]))
    if t in BINARY:
        return (t, rat_tuple(e[1]), rat_tuple(e[2]))
    return e


def to_sp(e, names):
    t = e[0]
    if t == 'var':
        return sp.Symbol(names[e[1]])
    if t == 'const':
        v = e[1]
        if isinstance(v, Fraction):
            return sp.Rational(v.numerator, v.denominator)
        fv = float(v)
        if abs(fv - round(fv)) < 1e-9:
            return sp.Integer(int(round(fv)))
        return sp.Float(fv, 12)
    if t == 'pow':
        return to_sp(e[1], names) ** int(e[2])
    if t in UNARY:
        return _UFUNC[t](to_sp(e[1], names))
    if t in BINARY:
        return _BFUNC[t](to_sp(e[1], names), to_sp(e[2], names))
    raise ValueError(t)


def _clean_worker(tuple_expr, names, q):
    try:
        rf = rat_tuple(tuple_expr)
        e = to_sp(rf, names)
        e = sp.cancel(e)
        e = sp.simplify(e)
        # 直接传回 sympy 对象（可 pickle），不走 str：物理名可能与 sympy 内置
        # 同名（gamma/beta/zeta/I/E/N...），str->parse 会把它们误解析成函数/常量。
        q.put(('ok', e))
    except Exception as ex:
        q.put(('err', str(ex)))


def clean_sympy(tuple_expr, names, timeout=60):
    """tuple -> 高精度化简后的 sympy（带超时隔离；失败返回 None）。"""
    q = mp.Queue()
    p = mp.Process(target=_clean_worker, args=(tuple_expr, names, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return None
    try:
        status, payload = q.get_nowait()
    except Exception:
        return None
    if status != 'ok':
        return None
    return payload
