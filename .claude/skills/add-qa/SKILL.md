---
name: add-qa
description: 向 interview_qa.json 题库添加新面试题，自动从知识库生成 key_points 和 reference_answer，并验证格式。
disable-model-invocation: true
allowed-tools: Read, Edit, Bash, Grep, Glob
---

# 向题库添加新题目

需求描述：$ARGUMENTS

示例调用：
- `/add-qa 加一道关于 HyDE 的 hard 题`
- `/add-qa mcp tool-calling medium 难度`
- `/add-qa 补充 agent memory 相关的 easy 题`

---

## Step 1：分析需求，确定题目参数

从 $ARGUMENTS 中提取：
- **topic**：rag / mcp / agent
- **difficulty**：easy / medium / hard
- **知识点关键词**：用于检索知识库

若信息不完整，合理推断（不要反问）。

## Step 2：读取现有题库，了解格式与覆盖情况

读取 `rag/code/interview_qa.json`，关注：
- 已有 ID 的最大编号（新 ID 递增）
- 该 topic 下已覆盖哪些知识点（避免重复）
- 现有 key_points 的粒度和风格（新题保持一致）

## Step 3：从知识库中检索知识点

根据 topic 搜索对应文件：
- **rag 相关**：优先读 `rag/code/mock-interview/` 下的 md 文件，其次 `rag/docs/*.html`
- **mcp 相关**：读 `mcp/05-interview/qa.md`、`mcp/05-interview/common-misconceptions.md`
- **agent 相关**：读 `agent/agent-5d-v3.html`

用 Grep 搜索关键词定位具体段落，提取核心知识点。

## Step 4：生成新题目

严格按以下 JSON 格式生成（参考现有题目的风格）：

```json
{
  "id": "<topic>-<三位数编号>",
  "topic": "<rag|mcp|agent>",
  "difficulty": "<easy|medium|hard>",
  "question": "<清晰、具体的面试问题>",
  "key_points": [
    "<核心知识点1，1句话>",
    "<核心知识点2，1句话>",
    "<核心知识点3，1句话>"
  ],
  "reference_answer": "<2-4句完整参考答案，涵盖所有 key_points>"
}
```

**key_points 质量标准**：
- 3-5 条，每条独立、可单独判断
- 用陈述句，不用问句
- 粒度参考：`"RRF 只看排名不看原始得分"` 而非 `"RRF 原理"`

## Step 5：追加到题库并验证

1. 读取完整 `interview_qa.json`
2. 将新题追加到数组末尾
3. 用 Edit 工具更新文件（保持合法 JSON）
4. 验证格式：
```bash
python3 -c "import json; data=json.load(open('rag/code/interview_qa.json')); print(f'题库共 {len(data)} 道题，最新：{data[-1][\"id\"]}')"
```
5. 跑测试确保文件改动未破坏现有逻辑：
```bash
python3 -m pytest rag/code/tests/ -q
```

## Step 6：输出确认

打印新增题目的完整内容，让用户确认 key_points 是否准确。
