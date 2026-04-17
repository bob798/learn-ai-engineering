---
name: rag-eval
description: 分析 RAG 实验结果 JSON 文件，解释指标含义，识别异常，给出一句话结论。支持 v4-v9 的所有结果文件。
disable-model-invocation: true
allowed-tools: Read, Bash
---

# RAG 实验结果分析

目标文件：`$ARGUMENTS`（如 `v5_hybrid_result.json`，留空则分析所有结果文件）

---

## Step 1：读取文件

```bash
ls rag/code/*result*.json rag/code/baseline.json 2>/dev/null
```

若 $ARGUMENTS 非空，读取指定文件；否则读取全部结果文件。

## Step 2：识别指标并解释

根据 JSON 结构自动识别指标类型：

**检索指标**（retrieval）：
| 指标 | 含义 | 健康范围 |
|---|---|---|
| `recall@3` / `recall_at_3` | 前3个结果中是否包含相关文档 | > 0.8 |
| `mrr` | 相关文档排多靠前（倒数排名均值） | > 0.7 |
| `ndcg` | 考虑排名权重的综合检索质量 | > 0.7 |

**生成指标**（generation，来自 RAGAS）：
| 指标 | 含义 | 健康范围 |
|---|---|---|
| `context_recall` | LLM 回答覆盖了多少检索内容 | > 0.8 |
| `context_precision` | 检索内容有多少真正被使用 | > 0.5 |
| `faithfulness` | 回答是否忠实于检索内容（不幻觉） | > 0.7 |
| `answer_relevancy` | 回答与问题的相关程度 | > 0.8 |

## Step 3：异常检测

逐条检查：
- **整体低分**：哪个指标明显低于健康范围？
- **单题异常**：`details` 中是否有某道题得分特别低（< 0.3）？
- **版本对比**：若有多个文件，指标是否比上一版本提升？
- **不一致**：`recall` 高但 `faithfulness` 低 → 检索到了但 LLM 没用上

## Step 4：输出结论

格式：

```
## [文件名] 分析结果

**一句话结论**：[最重要的发现]

**指标详情**：
- ✅ recall@3: 0.92 — 检索覆盖率良好
- ⚠️ context_precision: 0.33 — 检索结果中有 2/3 是噪音，建议缩减 top_k 或加 reranker
- ❌ faithfulness: 0.50 — LLM 存在幻觉，考虑在 prompt 中加"只根据以下内容回答"

**最值得关注的问题**：[具体 query ID 和原因]

**下一步建议**：[针对最低分指标的改进方向]
```

若分析多个文件，最后输出版本对比表：

```
| 版本 | recall@3 | mrr | faithfulness | answer_relevancy |
|---|---|---|---|---|
| baseline | ... | ... | - | - |
| v5_hybrid | ... | ... | - | - |
| v8_eval | ... | ... | ... | ... |
```
