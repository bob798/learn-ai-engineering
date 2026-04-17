---
name: eval-debug
description: 从开发视角分析面试 JSONL 文件，评估模型可用性、评分准确性、程序逻辑正确性，输出 bug 报告和改进建议。
disable-model-invocation: true
allowed-tools: Bash, Read
---

# 评估系统调试分析

目标文件：`$ARGUMENTS`（留空取最新）

---

## Step 1：定位文件

```bash
ls -lt rag/code/logs/*.jsonl | head -5
```

## Step 2：运行诊断脚本

```bash
python3 - << 'EOF'
import json, sys
from pathlib import Path
from collections import defaultdict, Counter

log_path = "$ARGUMENTS" or sorted(
    Path("rag/code/logs").glob("*.jsonl"),
    key=lambda p: p.stat().st_mtime
)[-1]

records = []
for line in Path(log_path).read_text().splitlines():
    try:
        r = json.loads(line)
        if "question_id" in r:
            records.append(r)
    except:
        pass

# ── 1. 模型可用性 ──
print("=== 模型可用性 ===")
model_status = defaultdict(lambda: {"total": 0, "failed": 0, "errors": []})
for r in records:
    for e in r.get("model_evals", []):
        m = e["model"]
        model_status[m]["total"] += 1
        if e["score"] < 0:
            model_status[m]["failed"] += 1
            err = e.get("error", e.get("feedback", ""))[:60]
            if err not in model_status[m]["errors"]:
                model_status[m]["errors"].append(err)

for model, stat in model_status.items():
    fail_rate = stat["failed"] / stat["total"] * 100 if stat["total"] else 0
    status = "✗ 全部失败" if stat["failed"] == stat["total"] else \
             f"⚠ {stat['failed']}/{stat['total']} 失败" if stat["failed"] else "✓ 正常"
    print(f"  {model.split('/')[-1]:<20} {status}")
    for err in stat["errors"][:1]:
        print(f"    原因: {err}")

# ── 2. 主裁判健康度 ──
print("\n=== 主裁判（primary model）健康度 ===")
primary_failures = 0
for r in records:
    evals = r.get("model_evals", [])
    if evals and evals[0]["score"] < 0:
        primary_failures += 1

print(f"  主裁判失败次数: {primary_failures}/{len(records)}")
if primary_failures > 0:
    print(f"  ⚠ BUG: key_points_hit/missed/feedback 来自主裁判，主裁判失败时这些字段为空")
    print(f"  影响: 所有题目的定性分析（命中/遗漏要点）不可信")

# ── 3. 得分准确性审查 ──
print("\n=== 得分合理性审查 ===")
for r in records:
    qid = r["question_id"]
    user_ans = r["user_answer"][:80]
    agg_score = r["score"]
    max_s = r["max_score"]
    evals = r.get("model_evals", [])
    valid_evals = [e for e in evals if e["score"] >= 0]
    scores = [e["score"] for e in valid_evals]

    issues = []

    # 模型分歧
    if len(scores) >= 2 and max(scores) - min(scores) >= 2:
        issues.append(f"模型分歧大 {[f'{e[\"model\"].split(\"/\")[-1]}:{e[\"score\"]}' for e in valid_evals]}")

    # 仅1个有效模型
    if len(valid_evals) == 1:
        issues.append(f"仅1个模型有效，得分可信度低")

    # 得分为0但回答非空
    if agg_score == 0 and len(user_ans.strip()) > 10:
        issues.append("得分0但有实质回答，可能误判")

    # 有 errors 但 score 不为0
    primary = evals[0] if evals else {}
    for e in valid_evals:
        if e.get("errors") and e["score"] > 0:
            issues.append(f"{e['model'].split('/')[-1']} 标记了错误但仍给分，逻辑需确认")

    print(f"\n[{qid}] {agg_score}/{max_s}")
    print(f"  回答: {user_ans}")
    if issues:
        for iss in issues:
            print(f"  ⚠ {iss}")
    else:
        print(f"  ✓ 评估稳定")

# ── 4. 误判检测 ──
print("\n=== 潜在误判（False Positive errors）===")
for r in records:
    for e in r.get("model_evals", []):
        if e.get("errors") and e["score"] >= 0:
            print(f"  [{r['question_id']}] {e['model'].split('/')[-1]}")
            print(f"  回答: {r['user_answer'][:80]}")
            for err in e["errors"]:
                print(f"  标记错误: {err}")
            print()

# ── 5. 程序逻辑校验 ──
print("=== 程序逻辑校验 ===")
for r in records:
    evals = r.get("model_evals", [])
    valid = [e for e in evals if e["score"] >= 0]

    # 校验 aggregated score 是否等于 valid scores 均值
    if valid:
        expected = round(sum(e["score"] for e in valid) / len(valid), 1)
        actual = r["score"]
        if abs(expected - actual) > 0.15:
            print(f"  ⚠ [{r['question_id']}] 得分聚合异常: 期望={expected} 实际={actual}")

    # 校验 key_points_hit + key_points_missed == all key_points (仅当主裁判正常时)
    if evals and evals[0]["score"] >= 0:
        all_kp = set(r.get("key_points_hit", []) + r.get("key_points_missed", []))
        # 只能粗判，不强要求

print("  (得分聚合逻辑校验完成)")

EOF
```

## Step 3：输出结构化 Bug 报告

格式：

```markdown
## 模型可用性
| 模型 | 状态 | 失败原因 |

## 发现的 Bug
| 严重级别 | 问题 | 影响 | 建议修复 |
| P0 | 主裁判失败时 key_points_hit/missed 为空 | 定性分析不可信 | fallback 到第一个成功的模型 |

## 得分可信度
每题：得分 / 有效模型数 / 是否有分歧

## 误判风险
列出 errors 字段中可疑的"误判"条目

## 改进建议
优先级排序的修复清单
```
