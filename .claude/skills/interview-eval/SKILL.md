---
name: interview-eval
description: 分析面试 JSONL 记录，输出得分复盘、薄弱点诊断、复习建议。传入文件名或留空使用最新记录。
disable-model-invocation: true
allowed-tools: Read, Bash, Glob
---

# 面试复盘分析

目标文件：$ARGUMENTS（若为空则自动取最新记录）

## Step 1：定位文件

若 $ARGUMENTS 为空：
```bash
ls -t rag/code/logs/*.jsonl | head -1
```
否则直接读 `rag/code/logs/$ARGUMENTS`。

## Step 2：读取并解析记录

逐行读取 JSONL，提取每道题的：
- question_id、topic、difficulty
- user_answer
- score / max_score
- key_points_hit / key_points_missed
- errors（评估器检测到的错误陈述）
- model_evals（各模型得分，若有）
- feedback

## Step 3：输出复盘报告（结构固定）

### 总览
- 总得分 / 总分、百分比
- 按 topic 分组得分（rag / mcp / agent）
- 按 difficulty 分组得分（easy / medium / hard）

### 逐题分析
每道题输出：
```
[题号] Q: <问题前50字>
得分：X/Y  难度：<difficulty>
✓ 答对：<key_points_hit>
✗ 遗漏：<key_points_missed>
⚠ 错误陈述：<errors>（若有）
💬 评估反馈：<feedback>
```

### 评估质量审查
检查是否有明显误判（常见问题）：
- errors 里的内容是否真的错了？还是评估器幻觉？
- key_points_missed 里是否有候选人用不同措辞说过但被漏判的点？
- 多模型得分差异 > 1 分的题目，说明评估不稳定，标注出来

### 诊断 & 建议
1. **最薄弱的 3 个知识点**（得分最低的 key_points_missed）
2. **优先复习路径**：指向 rag/code/mock-interview/ 或 rag/docs/ 中的具体文件
3. **下次面试策略**：哪些 topic/difficulty 需要重点练习
