---
title: Spring AI 2.0 GA 升级实录——变了什么，还有哪些坑要填
description: 一次 2.0.0-M4 → 2.0.0 GA 的真实升级。最疼的不是 OpenAiApi 被删，而是 HTTP 客户端统一到 okhttp 后“一个超时旋钮管两种行为”，逼出了拆 bean。附上升级带来的全部变化清单和留下的技术债。
spring-ai: 2.0.0
status: draft
---

# Spring AI 2.0 GA 升级实录——变了什么，还有哪些坑要填

> 姊妹篇《OpenAI 模块被整个重写，我只改了一个配置类》讲了哪些设计接住了升级。这篇讲**真正疼的地方**——那些编译不报错、测试不一定覆盖、却会在生产把你放倒的变化，以及升级后留下的债。

升级背景同前篇：一个 ReAct 风格的 Spring AI 智能体后端，`2.0.0-M4 → 2.0.0` GA。

---

## 一、升级带来了哪些变化

### 1. 工具执行彻底外部化
所有 ChatModel 移除了内部工具循环，`internalToolExecutionEnabled` 被删。要么用 `ChatClient + ToolCallingAdvisor`（自动注册），要么自己用 `ToolCallingManager.executeToolCalls()` 手动跑。

### 2. OpenAI 模块重建在官方 `openai-java` SDK 上
和 Anthropic 同款处理。`OpenAiApi` 删除；连接参数（baseUrl / apiKey / timeout / maxRetries）**搬到了 `OpenAiChatOptions`**；`.defaultOptions` → `.options`；`.retryTemplate` → `.maxRetries`；`OpenAiChatModel.builder().build()` 在缺 client 时**从 options 自动建**（字节码里是 `Objects.requireNonNullElseGet(openAiClient, supplier)`，所以根本不用手写 `OpenAiSetup`）。

> 这条官方 upgrade-notes 主要写了 Anthropic，OpenAI 的同款重写没怎么强调。**别只读 release note，对着 jar `javap` 一遍才是事实。** 我就是编译报 `程序包 org.springframework.ai.openai.api 不存在` 才发现的。

### 3. HTTP 客户端统一到 okhttp —— 最隐蔽的一条
M4 时，阻塞 `.call()` 走 RestClient、流式 `.stream()` 走 WebClient，**可以各设各的超时**。GA 把两条路统一到一个 okhttp 客户端，`.call()` 和 `.stream()` **共用同一个 `callTimeout`**。

`javap` 确认：`SpringAiOpenAiHttpClient.newCall(...)` 每个请求都用 okhttp 的 `callTimeout`，而 callTimeout 限制**整段调用（含读完响应体）**。

后果：**给流式设一个短的整体超时，会把长回答按整段时长静默切断。** 我有一个无工具的 ChatModel，既给真 token 流用、又被标题生成/连接预热的阻塞 `.call()` 复用——M4 时靠两套底层客户端各自调参没事，GA 下一个旋钮兼顾不了。

只能**按用途拆 bean**：

```java
// 阻塞类（标题/预热）：要 60s 上限兜住网关挂起
@Bean noToolBlockingChatModel → .timeout(Duration.ofSeconds(60))
// 真 token 流：绝不能设短超时，否则 callTimeout 切断长答案
@Bean finalAnswerStreamChatModel → .timeout(Duration.ofMinutes(10))
```

### 4. URL 约定变了
官方 SDK 的 baseUrl **自带 `/v1`**（默认 `https://api.openai.com/v1`），service 只在后面拼 `chat/completions`，即 `{baseUrl}/chat/completions`。`completionsPath` 概念取消。

所以所有 base-url 配置都要**补上 `/v1`**——dev/prod 配置、健康检查、以及 wire 测试基类（这条最容易漏，漏了整套 wire 测试的 stub 全 miss）。

### 5. 基线变更（M4 已吸收，列出备查）
Jackson 3（`com.fasterxml.jackson` → `tools.jackson`）、Spring Boot 4 / Framework 7 / JSpecify。

### 6. 没用到但值得知道的
MCP 包重命名、advisor/memory 顺序调整 + conversationId 必填、options 严格不可变（`copy()` → `mutate()`）、Anthropic 也换官方 SDK。

---

## 二、还有哪些坑要填（升级暴露的债）

| 优先级 | 问题 | 说明 / 建议 |
|---|---|---|
| 高 | **流式超时是脆弱约定** | “流式 bean 不设短超时”全靠人记。未来有人给流式 bean 设了 60s → 长答案被 callTimeout 静默切断。**应加一条断言/测试守住**（流式 bean 的 timeout ≥ 阈值），把口头约定变成代码护栏。 |
| 高 | **未验证项必须在合并前跑** | 全套容器化集成测试、真网关的 `/v1` 路径 + 流式不切断 + 阻塞超时、Jackson 3 的 JSON 序列化回归。换底层 SDK 的“行为变没变”，编译过了不代表对。 |
| 中 | **超时语义退化** | 旧代码区分 connect(10s)/read(60s)，GA 单个 `.timeout()` 只映射到 `request`(=callTimeout)，connect 默认约 1min。要保留双值得用完整 `com.openai.core.Timeout`（它有 connect/read/write/request 四个字段）。 |
| 中 | **重试非严格等价** | 旧 `RetryTemplate`（退避策略 + 异常分类）→ SDK `maxRetries`（只是次数）。次数能对上，但退避、哪些异常算可重试不保证一致——用 timeout/5xx/429 的 mock 场景验证。 |
| 中 | **版本自管的债** | 如果你像我一样用本地 BOM override 钉版本（绕过公司统一父 BOM），那今后 Spring AI 的升级**不会自动跟随**，需要一个主动跟踪发版的流程。 |
| 低/后续 | **可考虑拥抱 GA 原生能力** | GA 的 composable `ToolCallingAdvisor` / ChatClient 也许能替掉自定义 ReAct 循环——评估“减自定义代码” vs “保留 hard-cap/审批/幂等定制”的取舍，别急着这次做。 |

---

## 三、一条值得单独记的反模式教训

**当框架统一了底层传输（RestClient + WebClient → 单一 okhttp），你过去依赖“两条传输各自调参”实现的两种行为，会悄悄塌缩成一个旋钮。**

升级时要专门排查一个问题：**“我有没有靠两个底层实现的差异，来实现两种行为？”**

我这次的“阻塞要超时、流式不要超时”就是典型。这类耦合：
- 编译不报错（API 都在）
- 测试不一定覆盖（要构造长流才暴露）
- release note 不会提（它只列 API 变更，不列“两个客户端合并了”这种实现细节）

最隐蔽，也最该在升级清单里专门留一行。

---

## 方法论附录：怎么把“破坏面”查干净

别只靠编译——一个 import 失败（如 `OpenAiApi`）会掩盖同文件后续错误，main 编译不过还会让 test 全量级联报“找不到符号”，真信号被噪声淹没。

我用的是两步：
1. **存在性核验**：把项目用到的每个 Spring AI 符号，对 GA jar 的 class 索引做集合比对（查删除/迁包）。
2. **签名核验**：对高频/高风险类逐个 `javap` 比对方法和构造器签名（查方法级破坏，比如 `internalToolExecutionEnabled` 是不是真没了、消息构造器有没有变）。

`javap` 才是事实来源。这套方法让我在动手前就知道：要改的只有“删 flag + 重写 model 构建 + base-url 补 /v1”，其余一律不碰。
