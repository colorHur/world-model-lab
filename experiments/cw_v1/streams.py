# -*- coding: utf-8 -*-
"""随机子流：一个根种子展开成互不重叠、可枚举 ID 的子流。

★ 为什么不用 `seed + 7919 * k`（旧仓库 `ChannelAdapter` 的做法）
    加法偏移在 k 变大时会**碰撞**，而且无法证明两段互不相关。
    这里用 `np.random.SeedSequence(root).spawn(n)` —— 官方保证 spawn 出的子序列
    在统计上独立，并且每个子序列有自己的 `spawn_key`（可打印的唯一 ID）。

★ 为什么按 (split, episode) 分层再 spawn
    v4 §7：`train/calibration/dev/test` 四份**完整轨迹**，由根 SeedSequence 及
    split / episode / plant,channel,noise 子流生成，保存唯一 ID；不再用 seed+episode 拼凑。
    这里把它落成三级：
        root ─┬─ split=train      ─┬─ episode=0 ─┬─ plant
              │                    │             ├─ channel
              │                    │             ├─ meas     (任务观测噪声)
              │                    │             ├─ csi      (探测测量噪声)
              │                    │             └─ packet   (发包成败)
              ├─ split=calibration ─┴─ ...
              ├─ split=dev        ...
              └─ split=test       ...

★ 关键设计：**外生流与策略无关**
    `plant` / `channel` / `packet` 三条子流在同一 (split, episode) 下对所有策略**逐位相同**
    —— 因此「同一轨迹配对」是构造上成立的，不是靠事后对齐。
    策略的动作会改变**哪些时刻使用**这些流，但不会改写**未来**的流
    （子流按需取值，且每条流各自独立）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPLITS = ("train", "calibration", "dev", "test")
STREAM_NAMES = ("plant", "channel", "meas", "csi", "packet")


@dataclass(frozen=True)
class StreamIDs:
    """一个 episode 的五条子流的可打印唯一 ID。"""

    split: str
    episode: int
    ids: dict[str, list[int]]

    def flat(self) -> dict[str, str]:
        return {k: "-".join(str(x) for x in v) for k, v in self.ids.items()}

    def as_record(self) -> dict:
        return {"split": self.split, "episode": self.episode,
                "spawn_keys": self.flat()}


def make_episode_rngs(
    root_seed: int,
    split: str,
    episode: int,
    split_index: dict[str, int] | None = None,
) -> tuple[dict[str, np.random.Generator], StreamIDs]:
    """返回 (5 条独立 Generator, 可打印的 StreamIDs)。

    Parameters
    ----------
    split_index
        split 名 → 整数编号。**未给出的 split 一律拒绝**——避免"随手加一个 split
        却与前一个共用子流"这种静默错误。默认使用 `SPLITS` 的顺序。
    """
    idx = dict(split_index) if split_index is not None else {s: i for i, s in enumerate(SPLITS)}
    if split not in idx:
        raise ValueError(f"未注册的 split={split!r}；已注册 {sorted(idx)}")
    if not isinstance(episode, (int, np.integer)) or episode < 0:
        raise ValueError(f"episode 必须是非负整数，收到 {episode!r}")

    seq = np.random.SeedSequence(entropy=int(root_seed), spawn_key=(int(idx[split]), int(episode)))
    children = seq.spawn(len(STREAM_NAMES))
    rngs = {name: np.random.Generator(np.random.PCG64(c)) for name, c in zip(STREAM_NAMES, children)}
    ids = StreamIDs(split=split, episode=int(episode),
                    ids={name: list(map(int, c.spawn_key)) for name, c in zip(STREAM_NAMES, children)})
    return rngs, ids


def stream_uniqueness_report(root_seed: int, plan: dict[str, int]) -> dict:
    """枚举 `plan = {split: n_episodes}` 下所有子流 ID，检查两件事：
    ① 全部唯一；② 同一 (split, episode) 重复构造得到**同一** ID（确定性）。

    返回的 dict 直接进证据 JSON —— 实验卡反例 8 的验收物。
    """
    seen: dict[str, str] = {}
    dup: list[str] = []
    deterministic_fail: list[str] = []
    total = 0
    for split, n in plan.items():
        for ep in range(int(n)):
            rngs, sid = make_episode_rngs(root_seed, split, ep)
            rngs2, sid2 = make_episode_rngs(root_seed, split, ep)
            if sid.as_record() != sid2.as_record():
                deterministic_fail.append(f"{split}/{ep}")
            for name, key in sid.flat().items():
                total += 1
                tag = f"{split}/{ep}/{name}"
                if key in seen:
                    dup.append(f"{tag} == {seen[key]}")
                else:
                    seen[key] = tag
    return {
        "root_seed": int(root_seed),
        "plan": plan,
        "n_streams": total,
        "n_unique": len(seen),
        "duplicates": dup,
        "determinism_failures": deterministic_fail,
        "pass": (not dup) and (not deterministic_fail) and (len(seen) == total),
    }


def cross_split_independence_probe(root_seed: int, n: int = 2000) -> dict:
    """抽样检查两条子流的经验相关性接近 0（快速健康检查，不是统计证明）。

    只报告数值，不做"通过/失败"判定 —— 单次抽样的相关本来就会偏离 0，
    把它设成闸门会让闸门变成噪声发生器。
    """
    out = {}
    a, _ = make_episode_rngs(root_seed, "train", 0)
    b, _ = make_episode_rngs(root_seed, "dev", 0)
    for name in STREAM_NAMES:
        x = a[name].standard_normal(n)
        y = b[name].standard_normal(n)
        out[name] = float(np.corrcoef(x, y)[0, 1])
    return {"n_samples": n, "corr_train0_vs_dev0": out,
            "note": "单次抽样数值，非统计检验；仅作健康检查"}
