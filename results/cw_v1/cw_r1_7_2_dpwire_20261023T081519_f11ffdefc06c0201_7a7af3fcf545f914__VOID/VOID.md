# VOID —— 这一份**不是**本轮证据，不要引用

原因两条（都在 `docs/CW-R1.7.2-DP接线交付.md` §5 记着）：

1. `verdict.json` 里的 `DP-WIRING-REJECTED` 是**误报**：`_settle()` 用
   `r["name"].split(" ")[0]` 取闸名，而 `W1` 的名字是 `W1（链上 G8d[dp]）…`（中文全角括号前没有空格），
   取不到键 ⇒ `W1` 被当成 FAIL。已改成 `^W1(?!\w)` 精确匹配。
2. 这次运行在写 `meta.json` 时崩了（`int64` 不可 JSON 序列化），**没有落 meta**（无 git head / src hash / 时间）。

**正确证据**：同批的 `cw_r1_7_2_dpwire_20261023T081531_…_70f2361f77c9f44c/`。
