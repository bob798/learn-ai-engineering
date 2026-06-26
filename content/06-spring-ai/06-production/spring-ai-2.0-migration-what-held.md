---
title: OpenAI 模块被整个重写，我只改了一个配置类——Spring AI 2.0 升级里那些“接住”变化的设计
description: Spring AI 2.0 GA 把 OpenAI 模块换成官方 openai-java SDK，OpenAiApi 直接删除。听起来要伤筋动骨，实际改动却收敛在一个配置类。复盘是哪几条设计决策提前站在了框架演进的方向上。
spring-ai: 2.0.0
status: draft
---

# OpenAI 模块被整个重写，我只改了一个配置类

> 背景：我维护一个基于 Spring AI 的智能体后端——ReAct 风格的工具调用 agent，跑在 Spring Boot 4 / Java 25 上，模型走一个 OpenAI 兼容网关（后端接 DeepSeek）。
>
> 从 `2.0.0-M4` 升到 `2.0.0` GA 时，我做的第一件事是把项目用到的**每一个 Spring AI 符号**对 GA 的 jar 逐个 `javap` 核验。结论吓了我一跳又让我松了口气：**36 个 import 里只有 1 个被删**，而那个删除（`OpenAiApi`）背后是“整个 OpenAI 模块换成官方 SDK”的大重写。

按理说“provider 模块换底层 SDK”是地动山摇级别的破坏。但实际落地，代码改动集中在**一个配置类 + 几处测试**。复盘下来，是几条当初不起眼的设计决策，提前接住了这次升级。

---

## 决策一：自己驱动 ReAct 循环，从不依赖框架的“内部工具循环”

Spring AI 的 ChatModel 早期版本里，工具调用可以让模型“内部递归”——框架替你把 tool_call 跑掉再喂回模型。我一开始就关掉了它（`internalToolExecutionEnabled=false`），改用自己的 `ReActEngine` + `ToolCallingManager.executeToolCalls(...)` 手动跑循环。理由很朴素：我要在每一轮之间插入 hard-cap、超时、审批、幂等这些治理逻辑，框架的黑盒循环塞不进去。

结果 GA 干脆把“模型内部自动执行工具”**整个删了**——2.0 起工具执行永远在框架外部。

于是这个号称 breaking 的变更，对我是个 **no-op**：

```java
// 删掉这一行即可，行为完全不变
OpenAiChatOptions.builder()
    .toolChoice("required")
-   .internalToolExecutionEnabled(false)   // GA 已移除：执行本来就永远在外部
    .build();
```

而我循环的核心——`executeToolCalls(Prompt, ChatResponse)`——签名一行没变。

> **takeaway**：框架的“自动魔法”（隐式循环、自动执行）是最大的耦合点。把编排控制权握在自己手里，看着是“重复造轮子”，实际是把自己放在了框架演进的下游——框架后来也朝“外部执行”走了，我什么都不用改。

---

## 决策二：依赖抽象（ChatModel），而不是传输实现（OpenAiApi）

GA 里 36 个 import 只有 `org.springframework.ai.openai.api.OpenAiApi` 被删——它是一个 **provider 的传输实现类**。所有走抽象层的代码全活着：

| 层 | 例子 | GA 是否变 |
|---|---|---|
| 抽象接口 | `ChatModel` / `ChatResponse` / `Generation` / `Prompt` | ❌ 没变 |
| 工具抽象 | `ToolCallingManager` / `ToolCallback` / `ToolExecutionResult` | ❌ 没变 |
| 消息构造 | `UserMessage` / `AssistantMessage` / `ToolResponseMessage` | ❌ 没变（~150 处构造点全安全） |
| **传输实现** | **`OpenAiApi`** | ✅ **删除** |

`OpenAiApi.builder().baseUrl().apiKey()...` 这种直接捏底层客户端的写法，是唯一中招的地方。

> **takeaway**：provider 的实现类会被重写（OpenAI 这次和 Anthropic 一样换成了官方 SDK），但接口层（ChatModel/ChatResponse）是稳定契约。**贴着抽象编程，破坏面自动收窄。** 一个判断标准：你的业务代码 import 里出现 `...openai.api.*` 这种 provider 内部包，就是未来的迁移债。

---

## 决策三：版本敏感的装配，集中在一处

所有“模型 + 客户端”的构建，集中在一个 `AiChatConfig` 里（两三个 `@Bean`）。所以 `OpenAiApi → OpenAIClient` 这种换底层 SDK 的重写，只动这一个文件 + 复用它的几个测试，而不是散落全仓库。

GA 的新写法反而更简单——连接参数搬到了 options 上，`build()` 在你没给 client 时会**从 options 自动建**：

```java
// 旧：手捏 OpenAiApi 当传输层
OpenAiApi api = OpenAiApi.builder().baseUrl(url).apiKey(...).build();
OpenAiChatModel.builder().openAiApi(api).defaultOptions(opts)...build();

// 新：连接信息进 options，client 自动建
OpenAiChatOptions opts = OpenAiChatOptions.builder()
    .model(model).baseUrl(url).apiKey(key)
    .timeout(Duration.ofSeconds(60)).maxRetries(1)
    .toolCallbacks(callbacks).build();
OpenAiChatModel.builder().options(opts).toolCallingManager(tcm).build();
```

> **takeaway**：把“框架版本敏感的接线”（client/model 构建、provider 配置）圈进一个 config 层。升级时你只需要在一个地方打开手术。散落在 20 个文件里的 `OpenAiApi.builder()` 才是噩梦。

---

## 决策四 & 五：适配层 + 测试入口都“单一收口”

- **适配层**：工具、消息都过自己的适配器（`ToolCallbackAdapter`、`MessageMapping`），Spring AI 的类型不直接漏进业务逻辑。换框架版本时业务无感。
- **测试 wire 入口**：所有 wire-level 集成测试的 base-url 注入只在**一个基类**里。GA 改了 URL 约定（base-url 必须含 `/v1`，见姊妹篇），整套 wire 测试靠改一行就全适配。

> **takeaway**：单一收口是把双刃剑——集中意味着“容易漏看”（我这次差点漏改那个测试基类），但一旦定位，**一处修复覆盖全套**。分散的代价是永远修不干净。

---

## 一条有争议的：紧跟版本，把大迁移摊成小步

我升级前已经在 `2.0.0-M4`（一个里程碑预览版）。所以 2.0 最重的几个变更——Spring Boot 4 基线、Jackson 3、内部工具循环移除——在 M4 阶段就**持续小步**吸收完了。到 GA 只剩增量。

这条有争议：里程碑版在“生产”里跑本身是风险。但它确实把“一次性大跃迁”摊成了“持续小步”。**权衡题**：你是愿意每个里程碑改一点，还是攒到 GA 一次性大改一版？没有标准答案，但如果你选了紧跟，记得用 CI 兜住。

---

## 小结：什么样的设计“抗升级”

把这次复盘收敛成一句话——**抗升级的本质是“控制耦合点的位置”**：

1. 编排控制权握在自己手里（别用框架的隐式循环）
2. 依赖抽象接口，把 provider 实现类挡在适配层后面
3. 版本敏感的装配单一收口（一个 config 类）
4. 测试基础设施也单一收口

这些都不是为了“抗升级”才做的——它们本来就是好的工程实践（低耦合、可测试、关注点分离）。**抗升级只是好设计的副产品。** 反过来也成立：一次框架大版本升级，是检验你架构耦合度的最好体检。

> 姊妹篇：《Spring AI 2.0 GA 升级实录——变了什么，还有哪些坑要填》，讲这次升级真正“疼”的地方和留下的债。
