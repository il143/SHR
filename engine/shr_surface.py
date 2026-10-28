# -*- coding: utf-8 -*-
"""SHR 受控分解的"模拟器"：MLP 拟合连续曲面，可任意查询/受控扫描。

模型（MLP）只当算子——提供可任意采样的连续 f_hat；
SHR 判据在曲面上做受控实验（固定其他变量、只变一个）。
"""
import numpy as np
import torch
torch.manual_seed(0)


def _t(a, dev):
    return torch.tensor(np.ascontiguousarray(a).tolist(),
                        device=dev, dtype=torch.float32)


class Surface:
    def __init__(self, X, y, width=128, depth=4, epochs=6000, logy=False,
                 seed=0):
        torch.manual_seed(seed)
        self.X = np.asarray(X, float); self.y = np.asarray(y, float)
        self.logy = logy
        self.lo = self.X.min(0); self.hi = self.X.max(0)
        self.xm = self.X.mean(0); self.xs = self.X.std(0) + 1e-9
        if logy:
            tgt = np.log(np.abs(self.y))
        else:
            tgt = self.y
        self.ym = tgt.mean(); self.ys = tgt.std() + 1e-9
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        Xn = (self.X - self.xm) / self.xs
        yn = (tgt - self.ym) / self.ys
        layers = [torch.nn.Linear(X.shape[1], width), torch.nn.Tanh()]
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(width, width), torch.nn.Tanh()]
        layers += [torch.nn.Linear(width, 1)]
        net = torch.nn.Sequential(*layers).to(self.dev)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-6)
        Xt, yt = _t(Xn, self.dev), _t(yn, self.dev)
        for ep in range(epochs):
            net.train(); opt.zero_grad()
            l = torch.nn.functional.mse_loss(net(Xt).squeeze(1), yt)
            l.backward(); opt.step()
            if ep == int(epochs * 0.6):
                for g in opt.param_groups:
                    g['lr'] = 5e-4
        self.net = net
        # 拟合精度
        self.train_r2 = self._r2(self.y, self.query(self.X))

    def _r2(self, y, p):
        return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)

    def query(self, Xg):
        Xn = (np.asarray(Xg, float) - self.xm) / self.xs
        self.net.eval()
        with torch.no_grad():
            z = np.array(self.net(_t(Xn, self.dev)).squeeze(1).cpu().tolist())
        z = z * self.ys + self.ym
        if self.logy:
            z = np.exp(z) * np.sign(self.y.mean())
        return z

    def sweep(self, fixed, i, grid):
        m = len(grid)
        Xg = np.tile(np.asarray(fixed, float), (m, 1))
        Xg[:, i] = grid
        return self.query(Xg)

    def grid_i(self, i, m=24):
        return np.linspace(self.lo[i] + 0.02 * (self.hi[i] - self.lo[i]),
                           self.hi[i] - 0.02 * (self.hi[i] - self.lo[i]), m)
