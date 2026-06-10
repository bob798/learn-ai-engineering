# AI 记忆：短期 + 长期 动手 Demo

配套理论：[`content/03-rag/ai-memory-short-term-vs-long-term.md`](../../content/03-rag/ai-memory-short-term-vs-long-term.md)（译自 mem0.ai）

一组**纯内存、可单文件运行**的渐进式 demo，把「短期记忆 + 长期记忆 + 写穿透编排」从原理手搓一遍，最后对照 mem0 开源库（A 路线）。生产里换成 Redis / Qdrant 时，只有存储层实现要改，业务逻辑不变。

## 文件地图

| 文件 | 对应译文章节 | 是否需要 API/联网 | 一句话 |
|---|---|---|---|
| `00_配置提供商.py` | — | 测试连通性时需要 | 统一 `embed()` / `chat()` / `cosine_sim()`，换提供商只改 `.env` |
| `01_短期记忆_会话缓冲.py` | 《短期记忆》 | ❌ 纯逻辑，离线可跑 | 滑动窗口 + token 预算；演示朴素截断如何引发「灾难性遗忘」 |
| `02_长期记忆_向量存储.py` | 《长期记忆》 | ✅ 调 embedding | 内存向量库：写入 / 检索 Top-K / 行级隔离 / 陈旧更新 |
| `03_写穿透_记忆编排.py` | 《如何结合短期与长期》 | ✅ 调 chat + embedding | 写穿透五步 + LLM 事实抽取整合，手搓一个记忆智能体 |
| `04_用mem0库.py` | A 路线 | 安装 mem0ai 后可跑 | 同样的事，mem0 两行搞定（未装库则只打印说明） |

## 快速开始

```bash
cd code/memory
pip install -r requirements.txt
cp .env.example .env          # 填入 API Key，选 PROVIDER

# 1) 短期记忆——离线即可看效果，最快入门
python 01_短期记忆_会话缓冲.py

# 2) 验证模型连通性（需要 .env）
python 00_配置提供商.py

# 3) 长期记忆 / 写穿透编排（需要 .env + 联网）
python 02_长期记忆_向量存储.py
python 03_写穿透_记忆编排.py

# 4) A 路线：直接用 mem0 开源库
pip install mem0ai
python 04_用mem0库.py
```

国内推荐用**硅基流动（siliconflow）**：一个 key 同时给 embedding（BGE-M3）和 chat，有免费额度。

## 你会看到什么

- **01**：从对话头部砍消息 → 系统提示被删，模型忘记自己是谁；固定（pin）系统提示 + 滑动窗口 → 人设与红线始终在场。
- **02**：用户 A 检索不到用户 B 的记忆（行级隔离）；用户搬家后若不更新 embedding，模型会自信地答出旧地址（数据陈旧）。
- **03**：第 1 轮说「我叫 Sarah、只写 Python」被整合进长期记忆；第 2 轮推荐框架时这些事实被检索回来注入上下文——**跨轮、跨会话记忆生效**。

## 关于「A 路线 = 克隆 mem0」

mem0 是 **Apache-2.0** 开源，允许本地自建、修改、商用。`04_用mem0库.py` 演示把它的 `llm` / `embedder` 指向 OpenAI 兼容网关（如硅基流动）即可国内可用。

> ⚠️ 合规提醒：克隆**代码**自建合法；但若想做对外的**官网/文档中文镜像站**，会涉及文档版权与 "Mem0" 商标，需先取得授权——详见仓库根目录相关说明。本目录只做学习用途的自建与自用翻译。

## 从 Demo 到生产

| 这里（学习版） | 生产替换为 |
|---|---|
| 内存 list 当向量库 | Qdrant / Pinecone / pgvector |
| 同步 `consolidate()` | 异步 worker（Celery / 队列） |
| token 预算用字符数估算 | 真实 tokenizer（tiktoken 等） |
| 进程内会话状态 | Redis 会话存储 |
| 手搓编排 | mem0 / LangMem 等记忆层 |
