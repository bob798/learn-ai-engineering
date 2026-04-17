---
name: retro
description: 复盘近期对话，识别哪些重复操作可以做成 skill 或工作流，输出提效建议并保存到 pis/每周复盘/。
disable-model-invocation: true
allowed-tools: Read, Write, Bash, Glob
---

# 对话复盘 & 提效分析

范围：$ARGUMENTS（留空 = 当前项目近一周的所有会话）

---

## Step 1：定位会话文件

```bash
PROJECT_DIR=~/.claude/projects/-Users-bob-workspace-ai-handbook
ls -lt $PROJECT_DIR/*.jsonl | head -20
```

筛选一周内（近 7 天）修改过的 jsonl 文件。

## Step 2：提取有效内容

对每个会话文件，运行解析脚本：

```bash
python3 - << 'EOF'
import json, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

project_dir = Path.home() / ".claude/projects/-Users-bob-workspace-ai-handbook"
cutoff = datetime.now(timezone.utc) - timedelta(days=7)
results = []

for jf in sorted(project_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
    if jf.name == "session_log.jsonl":
        continue
    mtime = datetime.fromtimestamp(jf.stat().st_mtime, tz=timezone.utc)
    if mtime < cutoff:
        continue

    session = {"file": jf.name, "date": mtime.strftime("%Y-%m-%d"), "messages": [], "tools": []}
    with open(jf) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except:
                continue
            if rec.get("type") == "user":
                content = rec["message"].get("content", "")
                if isinstance(content, list):
                    content = " ".join(c.get("text","") for c in content if isinstance(c,dict))
                content = content.strip()
                if content and not content.startswith("["):
                    session["messages"].append(content[:200])
            elif rec.get("type") == "assistant":
                for blk in rec.get("message", {}).get("content", []):
                    if isinstance(blk, dict) and blk.get("type") == "tool_use":
                        session["tools"].append(blk.get("name",""))
    results.append(session)

for s in results:
    print(f"\n=== {s['date']} | {s['file'][:8]} ===")
    print("用户消息:")
    for m in s["messages"]:
        print(f"  - {m}")
    from collections import Counter
    tool_counts = Counter(s["tools"])
    print(f"工具调用: {dict(tool_counts)}")
EOF
```

## Step 3：识别提效机会

对提取的内容，分析：

**可以做成 skill 的信号：**
- 用户反复描述同一类任务背景（每次都要解释）
- 多步骤操作序列固定（每次做同样顺序的事）
- 需要查同一类文件或文档

**可以做成 hook 的信号：**
- 每次做 X 之后都会做 Y（自动触发候选）
- 手动验证/检查类操作（可自动化）

**值得记录为工作流文档的信号：**
- 超过 5 步的流程
- 涉及多个工具或外部系统

## Step 4：输出结构化建议并保存

输出格式：

```markdown
# 对话复盘 - [日期范围]

## 本周做了什么
- [按项目/主题分类]

## 发现的提效机会

### 建议新建的 Skill
| skill 名 | 触发场景 | 预计节省 |
|---|---|---|

### 建议新建或优化的 Hook
| 触发时机 | 执行内容 | 原因 |
|---|---|---|

### 可复用的模式
- [值得固化的思路或做法]

### 不值得自动化的
- [一次性操作，不要过度工程化]

## 下周建议
- 优先事项 1
- 优先事项 2
```

将报告保存到：
`/Users/bob/workspace/pis/每周复盘/[YYYY-WNN].md`（当前周次，追加到已有文件或新建）

## Step 5：更新 memory（如有跨对话价值的发现）

如发现新的用户偏好或工作模式，更新：
`/Users/bob/.claude/projects/-Users-bob-workspace-ai-handbook/memory/`
