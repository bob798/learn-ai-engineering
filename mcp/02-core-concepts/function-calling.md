# Function Calling：MCP 的底层机制

理解 Function Calling（FC），是理解 MCP Tools 工作原理的基础。两者的关系：**FC 是 LLM 的原生能力，MCP 是在 FC 之上的标准化工程规范**。

类比：FC : MCP = HTTP : REST

---

## Function Calling 是什么

LLM 原生支持的能力：当模型判断需要调用工具时，**不直接返回文本，而是输出结构化 JSON** 告诉宿主代码"我要调用哪个函数、用什么参数"。

### 完整的 FC 循环（谁干什么）

```
用户："上海今天天气怎么样？"
         │
         ▼
【你的代码】把 用户消息 + Tool定义 发给 AI 模型
         │
         ▼
【AI 模型】分析后输出结构化 JSON（只是"表达意图"）:
  {
    "type": "tool_use",
    "name": "get_weather",
    "input": { "city": "上海" }
  }
         │
         ▼  ← ✋ AI 在此停止，等待结果
【你的代码】读取 JSON，执行真实操作:
  result = weather_api.query("上海")
  → 返回: { temp: 22, weather: "晴" }
         │
         ▼
【你的代码】把结果传回给 AI（作为 tool_result）
         │
         ▼
【AI 模型】拿到数据，生成最终自然语言回答:
  "上海今天晴天，气温 22°C，适合出行。"
         │
         ▼
【用户】看到最终回答
```

### 最重要的认知

**AI 永远不直接执行任何操作。**

AI 只做两件事：
1. 决定调哪个 Tool、填什么参数（意图识别 + 参数推断，输出 JSON）
2. 拿到结果后生成自然语言回答

**真正执行操作的永远是你的代码。**

这个设计不是偶然的——如果 AI 能直接执行，它就能任意访问网络、文件系统、数据库，完全不可控。通过代码作为中间层，可以做权限检查、参数验证、速率限制、审计日志，AI 的能力始终在开发者掌控之内。

---

## AI 停止后，谁触发代码执行

是**你的代码自己在控制整个循环**，不是 AI 主动通知。

AI 就是一个普通的 HTTP 接口——你调用它，它返回结果，你的代码判断返回类型决定下一步：

```python
response = await anthropic.messages.create(
    model="claude-sonnet-4-5",
    messages=conversation,
    tools=available_tools
)

# 你的代码检查 stop_reason，自己决定下一步
if response.stop_reason == "tool_use":
    # 从返回值里提取 Tool 调用信息
    tool_call = next(b for b in response.content if b.type == "tool_use")
    
    # 你的代码执行对应函数（AI 完全不参与这一步）
    result = await execute_tool(tool_call.name, tool_call.input)
    
    # 把结果拼回对话，再次调用 AI
    conversation.append({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tool_call.id,
            "content": json.dumps(result)
        }]
    })
    # 再次调用 AI...

else:
    # AI 直接回答，展示给用户
    show_to_user(response.content[0].text)
```

AI 不持有连接，不主动推送。你的代码每次都要把完整的对话历史重新传给它。**主控逻辑始终在你的代码里。**

---

## FC 的前世今生

```
2020-2022 · 史前时代
  AI 只输出文本，开发者用正则表达式提取信息
  让 AI 回答 "CITY:上海 DATE:今天"，再解析字符串
  AI 稍微换个格式就崩——极其脆弱

2023年6月 · 里程碑
  OpenAI 发布 Function Calling
  GPT-3.5/4 首次支持结构化 Tool 调用
  AI 可以输出标准 JSON，不再需要解析自然语言
  AI 从"聊天机器人"走向"能做事的 Agent"的真正起点

2023年下半年 · 碎片化
  Claude、Gemini、各开源模型都加入 Tool Call 支持
  但每家的 JSON 格式、字段名、调用方式都不一样
  一个 Tool Server 无法跨模型复用——生态碎片化出现

2024年11月 · 标准化
  Anthropic 发布 MCP
  在 FC 之上加标准协议：统一服务发现、传输层、三类能力
  FC 是原子能力，MCP 是工程化标准

2025年 · 现在
  几乎所有主流模型和框架都支持 Tool Call
  竞争从"支不支持"转向"生态有多少工具、调用多准确"
  MCP 成为 AI 工具生态事实标准候选
  Google 推出 A2A（Agent-to-Agent）作为竞争方案
```

---

## FC 与 MCP 的关系

```
Function Calling（LLM 原生能力）
  │  ← AI 输出结构化意图，代码执行
  ├── MCP Tools（对 FC 的标准化封装）
  │     ← 统一协议、服务发现、跨模型复用
  ├── ReAct 模式（使用 FC 的推理范式）
  │     ← Reasoning（思考）+ Acting（调用 Tool）循环
  └── Agent Loop（多轮 FC 调用的完整循环）
        └── Multi-Agent（多个 Agent 互相调用）
```

MCP 在 FC 基础上增加了：
- **服务发现**：Client 运行时自动发现 Server 提供了哪些 Tool
- **跨框架复用**：同一个 Server 同时服务 Claude/GPT/开源模型
- **传输层抽象**：stdio（本地）/ SSE/HTTP（远程）可切换
- **Resources & Prompts**：FC 只有 Tool 的概念，MCP 新增了这两类
