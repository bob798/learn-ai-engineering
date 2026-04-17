#!/usr/bin/env python3
"""
Stop hook：占位脚本，保留 hook 入口但不做自动记录。

收获/思维提升需要主动反思才有价值，由两种方式触发：
  1. 面试中：每道题结束后输入一句收获，自动写入 pis/daily/
  2. 对话中：手动运行 /retro skill，系统性复盘整个会话

自动提取行为记录（命令/文件）对"收获"没有意义，已移除。
"""
import sys

# 避免 Stop hook 循环
import json
try:
    data = json.loads(sys.stdin.read())
    if data.get("stop_hook_active"):
        sys.exit(0)
except Exception:
    pass

sys.exit(0)
