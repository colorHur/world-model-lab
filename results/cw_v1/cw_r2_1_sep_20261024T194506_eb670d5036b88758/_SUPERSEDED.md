# SUPERSEDED（保留，不删不改）

- 作废原因：**闸 `B1` 首跑 FAIL** —— `model_policy.py:26` 的 `import hashlib` 不在
  `model_api.ALLOWED_ABS_IMPORTS` 白名单内。判断：**闸是对的，物该改**（不为过闸放松白名单）。
- 处理：`model_policy.py` 去掉 `hashlib`，把 `readout_digest()` 换成可读的 `readout_signature()`
  （累积和 + 首末值）；`r2_1_sep.py` 同步改名。**白名单一字节未动。**
- 本目录只跑到闸，面板未跑（按预注册 §4：闸未全过不看面板）。
- 重跑目录见同批 `cw_r2_1_sep_<UTC>_<src_hash>`（src_hash 不同）。
