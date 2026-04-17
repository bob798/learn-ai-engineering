# OMC 赋能产品经理设计音频客服系统

> 场景：产品经理利用 oh-my-claudecode（OMC）的多 Agent 编排能力，加速音频客服产品的设计全流程。

---

## OMC 产品通道 Agent

| Agent | 能力 | PM 怎么用 |
|---|---|---|
| **product-manager** | 需求拆解、PRD 生成、优先级排序 | 输入模糊想法，输出结构化需求文档 |
| **ux-researcher** | 用户画像、场景分析、竞品调研 | 分析音频客服的用户旅程和痛点 |
| **information-architect** | 信息架构、流程图、状态机设计 | 设计客服对话流转和状态机 |
| **product-analyst** | 数据分析、指标定义、ROI 测算 | 定义客服系统的核心指标和成本模型 |

辅助角色：**researcher**（深度调研）、**architect**（技术架构建议）、**writer**（文档输出）、**critic**（对抗式审核）

---

## 四阶段操作流程

### Phase 1：需求探索（Team 模式）

```
PM 输入：
"我要设计一个音频客服系统，面向电商售后场景，
 目标是替代 60% 的人工客服通话"

OMC 编排：
  ux-researcher  → 输出用户画像 + 场景地图 + 竞品分析
  product-analyst → 输出市场规模 + 成本对比（人工 vs AI）
  product-manager → 综合两者，输出需求优先级矩阵
```

Team 模式让三个 Agent **管线式协作**，前一个的输出自动流入后一个。PM 不需要分别跟每个 Agent 对话。

### Phase 2：架构设计（Team 模式）

```
PM 输入：
"基于上面的需求，设计客服系统的对话流程和角色架构"

OMC 编排：
  information-architect → 输出对话状态机 + 角色分工图
  architect             → 输出技术架构建议（ASR → NLU → Agent → TTS）
  product-manager       → 审核两者，标注产品风险和 MVP 边界
```

关键价值：PM 不懂技术细节，但 **architect Agent 会用 PM 能理解的语言**解释技术方案的取舍。

### Phase 3：PRD 生成（Autopilot 模式）

```
PM 输入：
"把前两轮的结论整理成 PRD，包含用户故事、验收标准、
 技术约束和 MVP 范围"

OMC 编排：
  product-manager → 自主生成完整 PRD
  （Autopilot 模式：需求明确，不需要多 Agent 协作）
```

### Phase 4：评审预演（Ralph 模式）

```
PM 输入：
"帮我预演评审会，找出 PRD 里的漏洞"

OMC 编排：
  product-analyst → 挑战指标定义（"60% 替代率怎么衡量？"）
  ux-researcher   → 挑战用户体验（"老年用户语音识别准确率？"）
  architect       → 挑战技术可行性（"实时语音延迟能控制在多少？"）
  critic          → 综合审核，输出风险清单
```

Ralph 模式的价值：**持续验证循环**。每个 Agent 都带着「挑毛病」的视角审核，PM 在真正上评审会之前就修补了漏洞。

---

## PM 典型命令示例

```bash
# 竞品调研
"帮我调研音频客服赛道，重点关注：
 1. 现有方案（讯飞/阿里云/AWS Connect）的优劣
 2. 用户最痛的 3 个问题
 3. AI 客服 vs 人工客服的成本结构"
→ OMC 自动调度 researcher + product-analyst + ux-researcher

# 对话流程设计
"设计一个客服对话的状态机，覆盖：
 问候 → 意图识别 → 问题处理 → 满意度确认 → 结束"
→ OMC 调度 information-architect 输出 Mermaid 流程图

# 对抗式审核
"这个 PRD 有什么漏洞？假设你是技术总监，你会问什么？"
→ OMC 调度 critic + architect 进行对抗式审核
```

---

## 效率对比：有 OMC vs 没有 OMC

| PM 工作环节 | 没有 OMC | 有 OMC |
|---|---|---|
| 竞品调研 | 自己搜索整理，2-3 天 | researcher Agent 输出结构化报告，2-3 小时 |
| 用户画像 | 找 UX 同事排期，1-2 周 | ux-researcher Agent 基于行业数据生成初版，PM 审核修改 |
| 对话流程设计 | 画 Visio/Figma 反复改，1 周 | information-architect 输出状态机，PM 调整业务规则 |
| PRD 撰写 | 写 3-5 天，格式和完整性参差 | product-manager Agent 生成结构化 PRD，PM 填充判断和决策 |
| 评审预演 | 找同事模拟，时间难协调 | critic Agent 随时可用，多视角对抗式审核 |
| 技术可行性判断 | 等研发评估，1-2 周 | architect Agent 即时给出技术约束和风险 |

---

## 核心价值

OMC 给 PM 的不是「替代思考」，而是三个加速器：

1. **调研加速** —— researcher / ux-researcher / product-analyst 把信息收集和结构化的脏活干了，PM 专注于**判断和决策**
2. **设计协作** —— information-architect / architect 提供专业视角，PM 不需要等排期就能获得多角色反馈
3. **质量防线** —— critic Agent 在评审前找漏洞，Ralph 模式持续验证，PM 带着更完善的方案上会

**一句话**：OMC 让 PM 拥有了一支随时待命的虚拟产品团队，从调研到 PRD 到评审预演，全流程都有专业角色配合。

---

*来源：ai-handbook L4 金丹境 · OMC 案例分析 · 2026-04-09*
