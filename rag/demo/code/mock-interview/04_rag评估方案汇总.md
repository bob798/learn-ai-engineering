# RAG 评估方案汇总

---

## 一、评估的三个阶段

```
文档 → [Chunking] → Embedding → [Retrieval] → Prompt → [Generation] → 回答
          ↑                         ↑                        ↑
       间接评估                   直接评估                  直接评估
```

---

## 二、核心指标体系

### Retriever 侧
| 指标 | 含义 | 怎么算 |
|------|------|--------|
| **Context Recall** | 该召回的都召回了吗 | 命中 ground truth 的比例 |
| **Context Precision** | 召回的里有多少是有用的 | 相关文档 / 总召回文档 |
| **Hit Rate** | Top-K 里有没有正确文档 | 有=1，无=0 |
| **MRR** | 正确文档排第几 | 1/rank 的均值 |

### Generator 侧
| 指标 | 含义 | 说明 |
|------|------|------|
| **Faithfulness** | 回答是否忠实于检索内容 | 有没有超出上下文发挥 |
| **Answer Relevancy** | 回答是否切题 | 和问题的相关程度 |
| **Hallucination** | 凭空捏造了多少 | RAGChecker 的 claim 级别检测 |
| **Noise Sensitivity** | 被无关文档带跑了吗 | 塞入噪声文档后回答是否变差 |

---

## 三、主流工具选型

| 工具 | 定位 | 适用阶段 |
|------|------|---------|
| **RAGAS** | 综合评估，最流行 | 研究 / 快速验证 |
| **RAGChecker** | Claim 级诊断，找锅归属 | 生产排查 |
| **DeepEval** | 5指标，可追溯到 chunk 参数 | 工程优化 |
| **Arize Phoenix** | Embedding 可视化 | debug 召回失败 |

---

## 四、没有 ground truth 怎么办

真实场景往往没有标注数据，两种解法：

1. **合成数据**：用 LLM 从文档自动生成 QA 对作为 ground truth（RAGAS 内置支持）
2. **无参考评估**：只看 Faithfulness + Answer Relevancy，不依赖标准答案

---

## 五、MRR vs RAGAS 4 指标：什么阶段用什么

| 指标 | 衡量维度 | 需要 LLM？ | 适用阶段 |
|------|---------|-----------|---------|
| **MRR** | 检索排名（top-1 是否命中） | ✗ 纯规则 | 早期快速迭代（v3.5~v7） |
| **Context Recall** | 信息覆盖度（连续值 0-1） | ✓ 语义判断 | 全面评估阶段 |
| **Context Precision** | 检索噪音比例 | ✓ 语义判断 | 全面评估阶段 |
| **Faithfulness** | 生成是否有幻觉 | ✓ 事实核查 | 全面评估阶段 |
| **Answer Relevancy** | 生成是否切题 | ✓ 相关性判断 | 全面评估阶段 |

**MRR 的两个盲区**：
1. 不衡量生成质量——检索完美，LLM 仍可能产生幻觉
2. 不区分"召回了有用的"vs"召回了噪音"——只关心 top-1 是否命中

**使用策略**：
- 早期快速迭代 → MRR：无 LLM 成本，一秒出结果，适合调参循环
- 阶段性全面评估 → RAGAS 4 指标：覆盖检索+生成，成本高但完整

---

## 六、学习路径对应

| 你的进度 | 对应评估动作 |
|---------|------------|
| V1 最小 RAG（当前） | 肉眼对比有/无 RAG 的回答差异 |
| V2 分块策略 | 改变 chunk_size，对比 Context Recall 变化 |
| V3 混合检索 | 对比纯向量 vs 混合的 Hit Rate |
| V4 评估 pipeline | 接入 RAGAS，跑自动化指标 |
| 生产优化 | 接入 RAGChecker，定位具体失效点 |
