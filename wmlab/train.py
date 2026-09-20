"""监督训练循环（世界模型）。

★ 为什么把它从 `scripts/02_train_world_model.py` 抽到这里（2026-09-19）：
   X24–X26 需要在同一个进程里多次训练世界模型（等价性检验要训两个模型对比，
   调度实验要用同配置的模型）。如果继续把训练函数放在脚本里，
   就会出现"两份实现"——一旦有人只改了一处，两次实验的模型就不可比了。
   训练逻辑属于**库**，不属于某个脚本。

行为与抽取前的 02 脚本**逐行一致**（同一个 seed 下结果逐位可复现）。
"""

from __future__ import annotations

import torch


def train_world_model(model, tr, va, cfg, device, verbose=True):
    """监督训练：预测下一步观测。返回 loss 历史。

    Args:
        model: MLPWorldModel
        tr, va: (obs, act, obs_next) 元组（torch.Tensor）
        cfg: 配置 dict（读 cfg["train"] 与 cfg["seed"]）
    """
    tcfg = cfg["train"]
    obs, act, nxt = tr
    obs_v, act_v, nxt_v = va

    opt = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]),
                           weight_decay=float(tcfg.get("weight_decay", 0.0)))
    n = obs.shape[0]
    bs = int(tcfg["batch_size"])
    epochs = int(tcfg["epochs"])

    hist = {"train_total": [], "train_recon": [], "train_latent": [], "val_total": []}
    g = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]))

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        tot = rec = lat = 0.0
        nb = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            b_obs = obs[idx].to(device)
            b_act = act[idx].to(device)
            b_nxt = nxt[idx].to(device)
            loss, parts = model.loss(b_obs, b_act, b_nxt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if tcfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
            opt.step()
            tot += float(loss.item()); rec += float(parts["recon"].item())
            lat += float(parts["latent"].item()); nb += 1
        hist["train_total"].append(tot / max(nb, 1))
        hist["train_recon"].append(rec / max(nb, 1))
        hist["train_latent"].append(lat / max(nb, 1))

        model.eval()
        with torch.no_grad():
            vloss, _ = model.loss(obs_v.to(device), act_v.to(device), nxt_v.to(device))
        hist["val_total"].append(float(vloss.item()))

        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"  epoch {ep:4d}  train={hist['train_total'][-1]:.6f}  "
                  f"val={hist['val_total'][-1]:.6f}  (recon={hist['train_recon'][-1]:.6f} "
                  f"latent={hist['train_latent'][-1]:.6f})")
    return hist
