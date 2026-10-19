#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T19 / P0-1 自检：运行锁（schema 2）的三个反例 + 一条 --force 反例。

为什么要有这个文件
------------------
P0-1 修的是"锁没锁住"这件事。**"我改成了 O_EXCL" 不是证据**，证据只能是
"两个进程同时抢，只活一个"这类**能被错误实现搞 FAIL** 的检查（自证伪 #39）。

四个反例（每个都能让错误实现 FAIL）
----------------------------------
C1 并发抢占：8 个进程同时 acquire，成功数必须 == 1（其余 exit 3）。
   ★ 错误实现（先 status 再 write）在这里会出多个 exit 0；重复 5 轮，5/5 都必须是 1。
C2 错误 token：错 token 的 release/refresh 必须 exit 4，且锁**仍在**（不能被顺手删掉）。
   另含：全局旧文件 `.run.id`（schema 1 的所有权证明）**不再**能让任何进程释放锁。
C3 租约过期但持有者仍活着：pid 指向活进程 ⇒ 普通 acquire = 3，`--takeover-expired` **也** = 3；
   pid 确死后再 `--takeover-expired` = 0（恢复路径存在，但必须显式）。
C4 `--force` 不得释放活动锁：活动租约下 `--force --token=<正确>` 必须 exit 4。

用法
----
    python scripts/49_p0_lock_selfcheck.py [--lock-tool PATH] [--rounds 5]

锁工具在**研究空间**（`06-进度管理/自动推进/tools/lock.py`），不在本仓库，
所以本脚本用 `--lock-tool` / 环境变量 `WMLAB_LOCK_TOOL` 定位；找不到 ⇒ exit 5（环境缺失，
**不是** PASS，也不许静默跳过）。

所有操作都在临时目录（`WMLAB_LOCK_PATH`）里做，**绝不碰真锁**。
退出码：0=全过 / 2=有 FAIL / 5=环境缺失。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_TOOL = Path("D:/workbuddy/researchproject/06-进度管理/自动推进/tools/lock.py")

RESULTS: list[dict] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return bool(ok)


def run(tool: Path, *args: str, lock_path: Path, timeout: float = 60.0) -> tuple[int, str]:
    env = dict(os.environ)
    env["WMLAB_LOCK_PATH"] = str(lock_path)
    p = subprocess.run([sys.executable, str(tool), *args], capture_output=True, text=True,
                       env=env, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def acquire_ok(tool: Path, lock_path: Path, *extra: str) -> tuple[int, str, str]:
    code, out = run(tool, "acquire", "--purpose", "selfcheck", "--ttl-min", "5", *extra,
                    lock_path=lock_path)
    token = ""
    for line in out.splitlines():
        if line.startswith("[ACQUIRED]"):
            for part in line.split():
                if part.startswith("token="):
                    token = part.split("=", 1)[1]
    return code, token, out


# ---------------------------------------------------------------- C0（阴性对照）
SIM_SRC = r'''
import json, os, sys, time, uuid
from pathlib import Path
LOCK = Path(__file__).with_name(".run.lock")
GAP = float(os.environ.get("SIM_GAP", "0"))
if LOCK.exists():                       # ← schema 1 的"先 status"
    print("HELD"); sys.exit(3)
if GAP:
    time.sleep(GAP)                     # ← 检查与写入之间的窗口（真实代码里也有，只是更窄）
LOCK.write_text(json.dumps({"run_id": uuid.uuid4().hex[:12]}))   # ← 再 write
print("OK"); sys.exit(0)
'''


def c0_negative_control(rounds: int) -> None:
    """旧逻辑（先 status 再 write）必须能让 C1 的判据 FAIL，否则 C1 没有分辨力。"""
    print("\n== C0 阴性对照：用 schema 1 的'先检查再写'逻辑跑同一并发测试 ==")
    d = Path(tempfile.mkdtemp(prefix="locksim_"))
    sim = d / "old_lock_sim.py"
    sim.write_text(SIM_SRC, encoding="utf-8")
    n = 8
    try:
        for gap, tag in ((0.0, "C0a 无人工间隙"), (0.02, "C0b 插入 20 ms 间隙")):
            winners_all = []
            for r in range(rounds):
                env = dict(os.environ)
                env["SIM_GAP"] = str(gap)
                # 模拟器把锁放在自己旁边；用子目录隔离每一轮
                sub = d / f"sim_{tag}_{r}"
                sub.mkdir()
                (sub / "old_lock_sim.py").write_text(SIM_SRC, encoding="utf-8")
                procs = [subprocess.Popen([sys.executable, str(sub / "old_lock_sim.py")],
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          text=True, env=env) for _ in range(n)]
                codes = [p.wait(timeout=120) for p in procs]
                winners_all.append(codes.count(0))
            mx = max(winners_all)
            if gap == 0.0:
                check(f"{tag}：记录旧逻辑的真实并发行为（无人工放大）",
                      True, f"各轮成功数={winners_all}（>1 即旧实现当场暴露；==1 说明窗口太窄、"
                            f"概率性通过 ⇒ 不能拿'跑过了'当安全）")
            else:
                check(f"{tag}：C1 的判据必须能 FAIL 掉旧逻辑", mx > 1,
                      f"各轮成功数={winners_all} ⇒ 最多 {mx} 个进程同时'拿到'同一把锁"
                      f"（期望 >1；若 ==1 则 C1 无分辨力，本自检不成立）")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- C1
def c1_concurrent(tool: Path, rounds: int) -> None:
    print("\n== C1 并发抢占（8 进程同时 acquire，成功数必须 == 1）==")
    n = 8
    for r in range(rounds):
        d = Path(tempfile.mkdtemp(prefix=f"lockc1_{r}_"))
        lp = d / ".run.lock"
        env = dict(os.environ)
        env["WMLAB_LOCK_PATH"] = str(lp)
        procs = [subprocess.Popen([sys.executable, str(tool), "acquire", "--purpose", f"c1-{r}",
                                   "--ttl-min", "5"], stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, env=env)
                 for _ in range(n)]
        codes = [p.wait(timeout=120) for p in procs]
        winners = codes.count(0)
        check(f"C1.{r} 单轮唯一赢家", winners == 1,
              f"8 进程退出码={sorted(codes)} ⇒ 成功 {winners}（期望恰好 1）")
        check(f"C1.{r} 输家全部是 exit 3（被占）", all(c in (0, 3) for c in codes),
              f"非 0/3 的退出码={[c for c in codes if c not in (0, 3)]}")
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- C2
def c2_token(tool: Path) -> None:
    print("\n== C2 错误 token 被拒 + 旧全局 .run.id 不再是所有权证明 ==")
    d = Path(tempfile.mkdtemp(prefix="lockc2_"))
    lp = d / ".run.lock"
    code, token, out = acquire_ok(tool, lp)
    check("C2.0 取锁成功且拿到 token", code == 0 and len(token) == 32, f"exit={code} token_len={len(token)}")

    st, _ = run(tool, "status", lock_path=lp)
    info = json.loads(run(tool, "status", lock_path=lp)[1])
    rid = (info.get("lock") or {}).get("run_id")
    # 锁内不许出现 token 明文
    raw = lp.read_text(encoding="utf-8") if lp.exists() else ""
    check("C2.1 锁文件不含 token 明文（只存 sha256）",
          token not in raw and "token_sha256" in raw, "grep token 明文" if token in raw else "ok")

    code, _ = run(tool, "release", "--token", "0" * 32, lock_path=lp)
    still = lp.exists()
    check("C2.2 错 token 的 release 被拒（exit 4）且锁仍在", code == 4 and still,
          f"exit={code} lock_exists={still}")

    code, _ = run(tool, "refresh", "--token", "f" * 32, lock_path=lp)
    check("C2.3 错 token 的 refresh 被拒（exit 4）", code == 4, f"exit={code}")

    code, _ = run(tool, "release", lock_path=lp)          # 完全不给 token
    check("C2.4 无 token 的 release 被拒（exit 4）", code == 4 and lp.exists(), f"exit={code}")

    # 旧 schema 1 的"所有权证明"：全局 .run.id。本条模拟"只拿到 run_id、没有 token"的会话
    (d / ".run.id").write_text(json.dumps({"run_id": rid}), encoding="utf-8")
    tok_file = d / ".run.tokens" / f"{rid}.json"
    if tok_file.exists():
        tok_file.unlink()
    code, _ = run(tool, "release", "--run-id", str(rid), lock_path=lp)
    check("C2.5 只有 run_id（旧全局 .run.id）不足以 release（exit 4，锁仍在）",
          code == 4 and lp.exists(), f"exit={code} lock_exists={lp.exists()}")

    code, _ = run(tool, "refresh", "--token", token, lock_path=lp)
    check("C2.6 正确 token 的 refresh 成功（exit 0，阴性对照）", code == 0, f"exit={code}")

    code, _ = run(tool, "release", "--token", token, lock_path=lp)
    check("C2.7 正确 token 的 release 成功（exit 0，阴性对照）", code == 0 and not lp.exists(),
          f"exit={code} lock_exists={lp.exists()}")
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- C3
def c3_expired_alive(tool: Path) -> None:
    print("\n== C3 租约过期但持有者仍活着 ⇒ 拒绝第二个写者 ==")
    d = Path(tempfile.mkdtemp(prefix="lockc3_"))
    lp = d / ".run.lock"
    code, token, _ = acquire_ok(tool, lp)
    check("C3.0 前置：取锁成功", code == 0, f"exit={code}")

    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(180)"])
    try:
        # 伪造"租约已过期，但 pid 是活进程"
        d0 = json.loads(lp.read_text(encoding="utf-8"))
        d0["expires_at"] = "2026-01-01T00:00:00+08:00"
        d0["pid"] = sleeper.pid
        lp.write_text(json.dumps(d0, ensure_ascii=False, indent=2), encoding="utf-8")
        info = json.loads(run(tool, "status", lock_path=lp)[1])
        check("C3.1 状态被判为过期", info["state"] == "STALE_EXPIRED", f"state={info['state']}")
        check("C3.2 且系统知道持有者 pid 还活着", info.get("holder_pid_alive") is True,
              f"holder_pid_alive={info.get('holder_pid_alive')}")

        code, out = run(tool, "acquire", "--purpose", "c3-plain", lock_path=lp)
        check("C3.3 普通 acquire 被拒（exit 3）", code == 3, f"exit={code}")

        code, out = run(tool, "acquire", "--purpose", "c3-takeover", "--takeover-expired",
                        lock_path=lp)
        check("C3.4 过期但持有者活着：--takeover-expired 也被拒（exit 3）", code == 3,
              f"exit={code} msg={out.strip().splitlines()[0] if out.strip() else ''}")
    finally:
        sleeper.kill()
        sleeper.wait(timeout=30)

    # 持有者确死 ⇒ 恢复路径必须存在，且必须显式
    d0 = json.loads(lp.read_text(encoding="utf-8"))
    d0["pid"] = 999999
    lp.write_text(json.dumps(d0, ensure_ascii=False, indent=2), encoding="utf-8")
    code, _ = run(tool, "acquire", "--purpose", "c3-nof", lock_path=lp)
    check("C3.5 过期+pid 已死：不带 --takeover-expired 仍拒绝（exit 3）", code == 3, f"exit={code}")
    code, tok2, out = acquire_ok(tool, lp, "--takeover-expired")
    check("C3.6 过期+pid 已死：--takeover-expired 可取锁（exit 0，恢复路径）", code == 0,
          f"exit={code}")
    if code == 0:
        run(tool, "release", "--token", tok2, lock_path=lp)
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- C4
def c4_force(tool: Path) -> None:
    print("\n== C4 --force 不得释放活动锁 ==")
    d = Path(tempfile.mkdtemp(prefix="lockc4_"))
    lp = d / ".run.lock"
    code, token, _ = acquire_ok(tool, lp)
    code, _ = run(tool, "release", "--force", "--token", token, lock_path=lp)
    check("C4.1 活动租约 + --force + 正确 token ⇒ exit 4 且锁仍在", code == 4 and lp.exists(),
          f"exit={code} lock_exists={lp.exists()}")
    code, _ = run(tool, "release", "--force", lock_path=lp)
    check("C4.2 --force 不带 token ⇒ exit 4（不误删他人锁）", code == 4 and lp.exists(),
          f"exit={code}")
    run(tool, "release", "--token", token, lock_path=lp)
    shutil.rmtree(d, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock-tool", default=os.environ.get("WMLAB_LOCK_TOOL", str(DEFAULT_TOOL)))
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()
    tool = Path(args.lock_tool)
    if not tool.exists():
        print(f"[SKIP] 找不到锁工具：{tool}\n"
              f"       用 --lock-tool 或 WMLAB_LOCK_TOOL 指定（它在本研究空间，不在本仓库）")
        return 5
    print(f"lock_tool = {tool}")
    print(f"python    = {sys.executable}")
    c0_negative_control(min(args.rounds, 3))
    c1_concurrent(tool, args.rounds)
    c2_token(tool)
    c3_expired_alive(tool)
    c4_force(tool)

    bad = [r for r in RESULTS if not r["ok"]]
    print(f"\n== 汇总：{len(RESULTS) - len(bad)}/{len(RESULTS)} PASS ==")
    for r in bad:
        print(f"  FAIL: {r['name']} — {r['detail']}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"checks": RESULTS, "n": len(RESULTS),
                                                   "n_fail": len(bad)}, ensure_ascii=False,
                                                  indent=2), encoding="utf-8")
    return 0 if not bad else 2


if __name__ == "__main__":
    sys.exit(main())
