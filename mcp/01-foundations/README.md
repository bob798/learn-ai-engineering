# MCP 基础：是什么、解决什么、为什么重要

## MCP 是什么

**Model Context Protocol（模型上下文协议）**，是 Anthropic 于 2024 年 11 月发布的开放标准，用于规范 AI 模型与外部工具/数据源之间的通信方式。

一句话版本：

> MCP 是给 AI 装插件的标准接口。你写一个 Server，声明"我有哪些工具"，任何支持 MCP 的 AI（Claude/Cursor/Windsurf）都能调用它们——写一次，全局复用。

---

## MCP 解决了什么问题

### 没有 MCP 时：N × M 的碎片化

每个 AI 应用对接每个外部系统，都要各自写一套对接代码。格式不统一，代码无法复用：

```
App1 对接 Slack  → 写一套代码
App1 对接 GitHub → 再写一套
App2 对接 Slack  → 再写一套（无法复用 App1 的）
App2 对接 GitHub → 再写一套
...
10 个 App × 20 个系统 = 200 套不可复用的集成代码
```

**关键误解澄清**：MCP 的 N+M 不是说"连接数从 200 减少到 30"，而是**开发工作量从 200 次变成 30 次**。物理连接还是会有，但每新增一个 App 的边际开发成本变成了零——因为 MCP Server 已经写好，直接复用。

### 有了 MCP 后：N + M 的标准化

```
20 个系统各自开发 MCP Server × 1 次 = 20 次工作
10 个 App 各自实现 MCP Client × 1 次 = 10 次工作
总计 = 30 次工作

Slack MCP Server 改了接口？ → 只改 Server 1 次，所有 App 自动生效
```

### N+M 背后的普世思想

这不只是一个技术方案，而是人类解决规模化问题的通用模式——**引入标准中间层，将乘法复杂度降为加法复杂度**：

| 领域 | 没有标准层 | 有标准层 |
|---|---|---|
| 经济 | 以物换物 N×N | 货币 N+N |
| 硬件 | 每种外设定制接口 | USB 标准 |
| 网络 | 私有通信协议 | HTTP |
| 数据库 | 私有查询语言 | SQL |
| AI 工具 | 各自写对接代码 | MCP |

---

## MCP 的三层架构

```
生态层（Ecosystem Layer）
  ↕ 飞轮效应、标准化价值、与其他协议的关系

应用层（Application Layer）
  ↕ Agent Loop、Tool 调用、多 Server 协作

协议层（Protocol Layer）
  Host / Client / Server 三角
  传输：stdio（本地）/ SSE/HTTP（远程）
  格式：JSON-RPC 2.0
```

> **命名建议**：不要叫"架构层"，容易和软件架构混淆。**协议层**更精确，聚焦于 MCP 的技术规范本身。

### Host / Client / Server 详解

| 角色 | 是什么 | 例子 |
|---|---|---|
| **Host** | 运行环境，持有 MCP Client | Claude Desktop、Cursor、你开发的 App |
| **Client** | Host 内部创建，**1:1 对应一个 Server** | 每个连接的 Server 对应一个 Client 实例 |
| **Server** | 暴露能力的服务 | GitHub MCP Server、你写的 file-server |

---

## MCP 的三类能力

| 能力 | 谁发起 | 有无副作用 | 本质 |
|---|---|---|---|
| **Tools** | AI 自主决定调用 | 有（执行动作） | 函数调用 |
| **Resources** | AI 按需读取 | 无（只读） | 数据访问 |
| **Prompts** | 用户主动触发 | 无 | 可复用提示词模板 |

详细说明见 [02-core-concepts/tools-resources-prompts.md](../02-core-concepts/tools-resources-prompts.md)

---

## 生态层：为什么说 MCP 有飞轮效应

```
Server 越多 → Host 接入价值越高
      ↑                ↓
Host 越多 ← Server 开发者意愿越高
```

主流 SaaS（Slack、GitHub、Google Drive）已陆续发布官方 MCP Server；
主流 Host（Claude Desktop、Cursor、VS Code）均已支持。

先发优势明显——就像 USB 出现后，外设厂商竞争点从"接口兼容性"转向"功能本身"，MCP 让 AI 应用的竞争从"连接哪些系统"转向"用得多好"。

---

## 延伸阅读

- [Host/Client/Server 架构详解](architecture.md)
- [MCP 与 REST/gRPC 的本质区别](../02-core-concepts/mcp-vs-rest-rpc.md)
- [MCP 的反对声音（批判性视角）](../04-advanced/criticisms.md)
