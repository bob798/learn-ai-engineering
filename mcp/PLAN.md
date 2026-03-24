# 仓库建设计划 · 执行路线图

## 现在就能发布（Phase 1）

以下内容已完成，可以立即推送到 GitHub：

### 已完成的文件

```
✅ README.md                           仓库主页
✅ 01-foundations/README.md            MCP是什么、三层架构、生态意义
✅ 02-core-concepts/
   ✅ tools-resources-prompts.md        三类能力详解（含大数据Resource场景）
   ✅ function-calling.md              FC前世今生、谁触发代码执行
✅ 03-practical/
   ✅ adapter-gateway.md               异构系统接入方案
✅ 05-interview/
   ✅ qa.md                            基础题+进阶题+实战题（完整版）
   ✅ common-misconceptions.md         学MCP时理解错的10件事（⭐SEO价值高）
✅ 06-my-perspective/
   ✅ presales-view.md                 从ToB预售视角看MCP
✅ interactive/
   ✅ mcp_overview.html                MCP总览交互笔记
   ✅ mcp_11q.html                     11个深度追问
   ✅ mcp_5q.html                      5个机制追问
   ✅ knowledge_methodology.html       5D知识习得方法论
✅ mcp-demo/                           Python MCP Server 实战代码（你的项目）
```

### Phase 1 发布前 To-Do（30分钟内完成）

- [ ] 替换 README.md 中的 `你的用户名` 和 `你的主页` 链接
- [ ] 替换 presales-view.md 中的主页链接
- [ ] 开启 GitHub Pages（Settings → Pages → Deploy from main）
- [ ] 测试 interactive/ 目录下 HTML 文件的 GitHub Pages 链接

---

## 下一步补充（Phase 2，1-2周内）

优先级排序：

### P1（最高价值）

- [ ] `03-practical/security-credentials.md`
  - OAuth 2.1 流程详解
  - 四种 credential 管理方案
  - Prompt Injection 防护

- [ ] `03-practical/large-data-resource.md`
  - Resource URI 设计模式
  - 代码库/知识库/数据库场景的具体实现

- [ ] `04-advanced/criticisms.md`
  - 有状态连接的问题
  - Prompt Injection 安全盲区
  - Tool 描述质量无法标准化
  - 与 Google A2A 的竞争

### P2（完善知识体系）

- [ ] `02-core-concepts/mcp-vs-rest-rpc.md`
  - 设计目标的根本差异（人读 vs AI 理解）
  - self-describing vs semantically opaque
  - 共存关系（不是竞争，是分层）

- [ ] `04-advanced/dynamic-tool-registration.md`
  - 实现逻辑代码示例
  - 三个核心价值
  - 四个真实弊端
  - 适用/不适用场景判断

- [ ] `04-advanced/tool-quality-evals.md`
  - Tool description 模板
  - LangSmith / PromptFoo 使用
  - 最小可行 Eval 体系

### P3（差异化内容）

- [ ] `06-my-perspective/speakeasy-mcp-design.md`
  - Speakeasy 的 MCP 架构设计
  - 为什么词典用 Resource 而不是 Tool
  - 学习模板的 Prompt 设计

- [ ] `06-my-perspective/presales-knowledge-base.md`
  - 用 MCP 打造预售知识库的设计草稿
  - Resource + Tool + Prompt 的完整设计

---

## Phase 3：内容联动（持续进行）

每次做 mcp-demo 的新功能，同步更新对应的文档：

| mcp-demo 功能 | 对应文档更新 |
|---|---|
| 新增一个 Tool | 03-practical/ 加对应设计说明 |
| 遇到 bug 或坑 | 05-interview/common-misconceptions.md |
| 性能优化 | 04-advanced/ 加对应最佳实践 |

---

## GitHub Pages 配置

```bash
# 1. 推送代码
git add .
git commit -m "feat: initial MCP handbook content"
git push origin main

# 2. 在 GitHub 仓库设置中开启 Pages
# Settings → Pages → Source: Deploy from branch → main / root

# 3. 交互式笔记的访问地址
https://你的用户名.github.io/mcp-handbook/interactive/mcp_11q.html
```

---

## 内容营销计划

发布仓库后，可以在以下平台发内容引流：

| 平台 | 内容形式 | 切入角度 |
|---|---|---|
| 视频号/Douyin | 短视频 | "我整理了一份可交互的 MCP 学习手册" |
| 知乎 | 长文 | 从 common-misconceptions.md 改写成知乎问答 |
| Bilibili | 讲解视频 | 用 HTML 文件作为讲解素材 |
| 小红书 | 图文 | "AI 工程师学 MCP 的笔记方法" |

**最强单篇**：`common-misconceptions.md` 改写成"学 MCP 时我理解错的 10 件事"，在知乎/掘金发，SEO 价值高，初学者容易搜到。
