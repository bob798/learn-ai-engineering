# OMC Ralph 模式深度拆解

> 日期: 2026-04-13 · 来源: oh-my-claudecode 源码分析 + claude-plugins-official/ralph-loop

## 一句话定义

**Ralph = 持久执行 + 自纠错 + 独立验证的循环模式。** 它的核心承诺是：任务不完成、不停止；完成了，也要经过独立审核才算完。

---

## 1. 起源与命名

Ralph 得名于 **Ralph Wiggum**（辛普森一家角色），由 [Geoffrey Huntley](https://ghuntley.com/ralph/) 首创。原始形态极其简单：

```bash
while true; do
  claude --prompt "$(cat prompt.md)"
done
```

核心哲学：**Iteration > Perfection**——不追求一次做对，靠循环迭代逼近正确。

### 真实战绩
- Y Combinator 黑客松中一晚自动生成 6 个完整仓库
- 一份 $50,000 合同用 $297 API 费用完成
- 用 3 个月创造了一门完整的编程语言（"cursed"）

---

## 2. 架构：两层实现

OMC 中的 Ralph 实际有**两层实现**，分别服务不同场景：

### 2.1 基础层：Stop Hook 循环（claude-plugins-official）

```
┌─────────────────────────────────────────┐
│  用户: /ralph-loop "任务描述"             │
│              ↓                           │
│  setup-ralph-loop.sh                     │
│  → 写入 .claude/ralph-loop.local.md      │
│  → 记录: active, iteration, max,         │
│          completion_promise, session_id   │
│              ↓                           │
│  Claude 执行任务                          │
│              ↓                           │
│  Claude 尝试退出                          │
│              ↓                           │
│  Stop Hook 拦截 ──→ 检查完成条件          │
│  │                                       │
│  ├─ iteration >= max? → 放行退出          │
│  ├─ 输出包含 <promise>完成词</promise>?   │
│  │   → 放行退出                          │
│  └─ 否则 → 阻止退出，回灌同一 prompt      │
│           iteration++, 继续循环           │
└─────────────────────────────────────────┘
```

**状态文件**: `.claude/ralph-loop.local.md`（YAML frontmatter + prompt 正文）

```yaml
---
active: true
iteration: 1
session_id: xxx
max_iterations: 50
completion_promise: "COMPLETE"
started_at: "2026-04-13T..."
---

Build a REST API for todos...
```

**关键设计**：
- **Completion Promise**：精确字符串匹配，必须用 `<promise>DONE</promise>` XML 标签包裹
- **反作弊机制**：prompt 中反复强调"MUST be completely and unequivocally TRUE"——防止模型为了逃出循环而撒谎
- **无限循环默认**：如果不设 `--max-iterations`，Ralph 会永远运行

### 2.2 编排层：Pipeline Stage Adapter（OMC 核心）

在 OMC 的 Autopilot Pipeline 中，Ralph 是一个**验证阶段适配器**：

```typescript
// src/hooks/autopilot/adapters/ralph-adapter.ts
export const ralphAdapter: PipelineStageAdapter = {
  id: 'ralph',
  name: 'Verification (RALPH)',
  completionSignal: 'PIPELINE_RALPH_COMPLETE',
  
  shouldSkip(config) {
    return config.verification === false;
  },
  
  getPrompt(context) {
    // 生成并行验证 prompt...
  }
};
```

**验证三审并行**：

| 审核 Agent | 模型 | 检查内容 |
|---|---|---|
| `architect` (功能完整性) | Opus | 所有需求已实现、验收标准满足、无遗漏 |
| `security-reviewer` (安全) | Opus | OWASP Top 10、输入验证、注入、密钥泄露 |
| `code-reviewer` (质量) | Opus | 设计模式、错误处理、测试覆盖、可维护性 |

**判决逻辑**：
- 每个 reviewer 输出 `APPROVED` 或 `REJECTED + 具体问题`
- **任一 REJECTED** → 收集所有拒绝原因 → 修复 → 重新验证
- **全部 APPROVED** → 发出 `PIPELINE_RALPH_COMPLETE` 信号

---

## 3. 状态管理

### 3.1 持久化状态

```
.omc/state/sessions/{sessionId}/
├── ralph-state.json              # 主状态
└── ralph-verification-state.json  # 验证阶段状态
```

**ralph-state.json**:
```json
{
  "active": true,
  "iteration": 4,
  "max_iterations": 10,
  "session_id": "xxx",
  "started_at": "2026-04-13T...",
  "prompt": "Implement issue #1496",
  "critic_mode": "codex"    // 可选：使用外部 critic
}
```

**ralph-verification-state.json**（验证阶段生成）:
```json
{
  "pending": true,
  "completion_claim": "All stories are complete",
  "verification_attempts": 0,
  "max_verification_attempts": 3,
  "requested_at": "2026-04-13T...",
  "original_task": "Implement issue #1496",
  "critic_mode": "critic"
}
```

### 3.2 PRD 驱动模式

Ralph 支持 **PRD（Product Requirements Document）驱动** 的结构化执行：

```json
// .omc/prd.json
{
  "project": "TestProject",
  "branchName": "ralph/test-feature",
  "description": "Feature description",
  "userStories": [
    {
      "id": "US-001",
      "title": "User authentication",
      "description": "As a user...",
      "acceptanceCriteria": ["JWT token", "Password hashing"],
      "priority": 1,
      "passes": false,
      "notes": ""
    }
  ]
}
```

**PRD 状态追踪**：
- `getPrdStatus()` → 计算完成率、下一个待办 story
- `markStoryComplete()` / `markStoryIncomplete()` → 更新 story 状态
- `getNextStory()` → 按 priority 排序返回下一个未完成 story
- 所有 stories 完成 → 触发 **验证阶段**（不是直接完成）

---

## 4. 安全机制

### 4.1 迭代上限

```
默认: 无限制
--max-iterations: 用户指定
Hard Max (OMC_SECURITY=strict): 200 次强制停止
Hard Max (默认): 500 次强制停止
```

当达到 max_iterations 但未达到 hard max 时，OMC 会**自动扩展**：
```typescript
// 源码逻辑: iteration达到100/100时
updated.max_iterations = 110;  // 自动+10继续
```

达到 hard max 时：
```typescript
updated.active = false;  // 强制停止，不可扩展
result.message = 'HARD LIMIT...';
```

### 4.2 Session 隔离

```typescript
// ralph-session-mismatch.test.ts 验证的场景
// session_id 不匹配时，Ralph 不会干扰其他会话
```

### 4.3 Critic 验证

Ralph 完成时不是自己说了算，而是要经过独立 Critic 验证：

```
PRD 全部完成
    ↓
进入 CODEX CRITIC VERIFICATION
    ↓
Critic 审查（可以是 codex/gemini/独立 critic agent）
    ↓
Critic 输出: <ralph-approved critic="critic">VERIFIED_COMPLETE</ralph-approved>
    ↓
Ralph 才真正结束
```

**这是 OMC "创作/审核分离" 原则在 Ralph 中的具体体现。**

---

## 5. HUD 可视化

```typescript
// src/hud/elements/ralph.ts
// 格式: ralph:3/10
// 颜色编码:
//   绿色: 正常迭代
//   黄色: 达到 warning 阈值
//   红色: 达到 90% max_iterations（即将触限）
```

---

## 6. Ralph 循环完整流程图

```
用户输入 "ralph: 重构 auth 模块"
         │
         ▼
    Magic Keyword 检测
    匹配 "ralph" → 激活 Ralph 模式
         │
         ▼
    ┌─ Ralph 循环开始 ──────────────────────┐
    │                                        │
    │  1. Executor 执行任务                   │
    │         │                              │
    │         ▼                              │
    │  2. 遇到错误?                           │
    │     ├─ 是 → 诊断 → 修复 → 回到 1       │
    │     └─ 否 → 继续                       │
    │         │                              │
    │         ▼                              │
    │  3. 任务完成?                           │
    │     ├─ 否 → iteration++ → 回到 1       │
    │     └─ 是 → 进入验证                   │
    │         │                              │
    │         ▼                              │
    │  4. 并行三审验证                        │
    │     ├─ Architect: 功能完整性            │
    │     ├─ Security Reviewer: 安全性        │
    │     └─ Code Reviewer: 代码质量          │
    │         │                              │
    │         ▼                              │
    │  5. 全部 APPROVED?                     │
    │     ├─ 否 → 修复问题 → 回到 4          │
    │     └─ 是 → Critic 终审               │
    │         │                              │
    │         ▼                              │
    │  6. Critic 验证通过?                    │
    │     ├─ 否 → 回到 1                     │
    │     └─ 是 → RALPH COMPLETE             │
    │                                        │
    └────────────────────────────────────────┘
```

---

## 7. 与其他模式的组合

| 组合 | 效果 | 适用场景 |
|---|---|---|
| `ralph autopilot` | Ralph 的持久性 + Autopilot 的完整工作流 | 全自动端到端开发 |
| `ralph ulw` (ultrawork) | Ralph 的持久性 + Ultrawork 的并行能力 | 大批量重构 |
| `ralplan` → `ralph` | 先共识规划再持久执行 | 需求不够明确的高风险任务 |

---

## 8. 适用 vs 不适用

### 适合 Ralph 的场景
- 复杂重构（可能遇到未知错误）
- 数据库迁移（需多步验证）
- 技术栈迁移（TypeScript/JWT/新框架）
- 有明确验收标准的功能开发
- 需要"走开去喝咖啡"的自动化任务

### 不适合 Ralph 的场景
- 简单问答或一次性操作
- 需要人类审美/设计判断的任务
- 没有明确成功标准的探索性任务
- 预算敏感场景（迭代消耗 Token 是单次的 3-10 倍）

---

## 9. 设计模式提炼

Ralph 体现了几个可迁移的 Agent 设计模式：

### 模式 1: Self-Referential Loop（自指循环）
- 同一个 prompt 反复执行，Agent 通过读取自己上次的文件输出来"记住"进度
- **迁移价值**: 任何需要多轮迭代的 Agent 任务（RAG 检索优化、代码生成-测试循环）

### 模式 2: Completion Promise（完成承诺）
- Agent 必须输出特定标记才能退出循环，且该标记必须"真实"
- **迁移价值**: 防止 LLM 过早声称完成——RAG 系统的回答质量门控

### 模式 3: Independent Verification（独立验证）
- 执行 Agent ≠ 验证 Agent，防止"自己审自己"
- **迁移价值**: RAG 系统的"生成回答 Agent" ≠ "审核回答 Agent"

### 模式 4: Graceful Degradation（优雅降级）
- 达到软上限 → 自动扩展；达到硬上限 → 强制停止
- **迁移价值**: 长时运行的 Agent 系统必须有成本/时间兜底

### 模式 5: PRD-Driven Execution（需求驱动执行）
- 用结构化的 User Stories + Acceptance Criteria 驱动迭代
- **迁移价值**: 将模糊的"做一个功能"转化为可追踪、可验证的任务分解

---

## 10. 一句话结论

**Ralph 不是"让 AI 重试"那么简单——它是一套完整的"持久执行 + 自纠错 + 独立审核 + 安全兜底"工程体系。** 其中最值得学习的不是 while-true 循环本身，而是三个深层设计决策：(1) Completion Promise 防止模型"撒谎完成"；(2) 创作/审核永远分离；(3) Hard Max 防止失控。这三个模式可以直接迁移到任何 RAG / Agent Memory / 自动化系统中。
