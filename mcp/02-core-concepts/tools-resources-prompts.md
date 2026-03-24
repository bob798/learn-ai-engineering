# Tools / Resources / Prompts：MCP 三类能力详解

MCP Server 可以暴露三类能力，很多人只知道 Tools，而忽略了 Resources 和 Prompts。三者设计意图完全不同。

---

## 对比总览

| 能力 | 谁发起 | 有无副作用 | 类比 | 适用场景 |
|---|---|---|---|---|
| **Tools** | AI 自主决定 | 有（写操作） | 函数调用 | 发邮件、查天气、创建订单 |
| **Resources** | AI 按需读取 | 无（只读） | 文件系统 | 读取文档、查询数据库 |
| **Prompts** | **用户**主动触发 | 无 | 斜杠命令 | 代码审查模板、学习计划生成 |

---

## Tools：AI 执行动作的能力

### 核心机制

Tool 是 MCP 最核心的能力。AI 决定是否调用、填什么参数，代码执行真实操作。

```python
# Server 侧定义一个 Tool
@mcp.tool()
def search_notes(query: str, limit: int = 10) -> list[dict]:
    """
    搜索本地 Markdown 笔记。
    [何时用] 用户需要查找特定主题的笔记时。
    [何时不用] 如果已知文件路径，用 read_file 更直接。
    [返回] 匹配的文件列表，含路径和摘要
    """
    # 真实的搜索逻辑
    return search_filesystem(query, limit)
```

### Tool Description 是质量瓶颈

MCP 标准化了接口格式，但无法标准化 description 的质量。**AI 选错 Tool 的根本原因几乎都是 description 写得不好**。

高质量 description 的结构：

```
[何时用]    明确的触发场景
[何时不用]  边界说明（防止和其他 Tool 混淆）
[输入]      参数的语义解释，不只是类型
[输出]      返回什么，格式是什么
[副作用]    会修改什么数据，发什么通知
[失败处理]  出错时返回什么，AI 应该怎么处理
```

---

## Resources：AI 按需读取数据

### 核心机制

Resource 是"数据地图 + 按需取数"。AI 先看目录（list），再精确读取需要的数据（read），不是一次性全部塞入 context。

```
# AI 的决策过程
list_resources()
→ file://src/auth.py      "用户认证模块"
→ file://src/db.py        "数据库连接层"
→ ... 共 500 个文件

# AI 自主判断只需要看 2 个文件
read_resource("file://src/auth.py")    # 消耗 ~1.5k tokens
read_resource("file://src/db.py")      # 消耗 ~1k tokens
# 总计: ~2.5k tokens（而非 500 个文件的 800k tokens）
```

### 大数据场景：Resource vs Tool 的选择原则

**当数据量超过 10k tokens 时，优先用 Resource 而不是 Tool 返回值。**

```
❌ Tool 方式（一次性塞入）
call_tool("get_codebase") → 返回全部 500 个文件 → 超出 context 限制

✅ Resource 方式（按需读取）
list_resources() → AI 判断只需要 3 个文件 → 精确读取
```

典型大数据场景的设计：

| 场景 | Resource 设计 |
|---|---|
| 代码库分析 | 每个文件一个 Resource，URI = 文件路径 |
| 企业知识库 | 每篇文档一个 Resource，Tool 做语义搜索返回 URI 列表 |
| 数据库查询 | 表 schema 是 Resource，实际查询是 Tool |
| 长文档处理 | 按章节拆成多个 Resource，目录是单独的 Resource |

### URI 寻址机制

每个 Resource 有一个唯一 URI，AI 可以精确请求：

```
file://project/src/auth.py
db://users/user_id_123
api://salesforce/opportunity/SF-001
log://app/2025-03-19/error
```

---

## Prompts：用户触发的模板能力

### 核心机制

Prompts 是三类能力里最容易被忽视的，但对产品化最有价值。

**区别于 Tools 和 Resources 的关键：Prompts 是用户主动触发的，AI 是被动的。**

触发路径：
```
Server 声明 Prompt 模板
    ↓
Host 展示为 "/" 斜杠命令（就像 Slack 的 slash command）
    ↓
用户选择并填入参数
    ↓
Host 调用 Server 展开模板（自动注入相关 Resource）
    ↓
完整 prompt 发给 AI，AI 开始执行
```

### 具体例子：代码审查模板

```python
@mcp.prompt()
def code_review(file_path: str, focus: str = "general") -> str:
    """
    对指定文件进行代码审查。
    focus 可选值: security / performance / style / general
    """
    file_content = read_file(file_path)
    
    focus_map = {
        "security": "重点检查：SQL注入、输入验证、权限控制、敏感数据处理、认证漏洞",
        "performance": "重点检查：N+1查询、内存泄漏、不必要的循环、缓存缺失",
        "style": "重点检查：命名规范、函数长度、注释质量、代码重复",
        "general": "全面审查代码质量、安全性、可维护性"
    }
    
    return f"""请对以下代码进行审查：

文件路径：{file_path}
审查重点：{focus_map.get(focus, focus_map['general'])}

代码内容：
{file_content}

请按以下格式输出：
1. 发现的问题（按严重程度排序）
2. 具体修改建议
3. 整体评估"""
```

用户在 Claude Desktop 输入 `/code_review`，填入 `file_path` 和 `focus` 两个参数，就能得到标准化的专家级代码审查——不需要用户自己会写好 prompt。

### Prompts 的产品价值

**把"专家用法"产品化**：普通用户填参数就能得到专家级结果，质量稳定可复用。

对比：

| 用户直接打字 | 使用 Prompt 模板 |
|---|---|
| "帮我看下 auth.py 的安全问题" | 选 `/code_review`，填 file_path + focus |
| AI 不知道要检查哪些维度 | 模板自动展开成含完整检查清单的专家 prompt |
| 质量取决于用户描述能力 | 质量稳定，Server 开发者保证 |

---

## Schema：参数定义规范

Schema 是参数结构的"说明书"——告诉 AI 这个参数是什么类型、什么含义、哪些必填、取值范围是什么。

```json
{
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "搜索关键词，支持姓名/邮件/手机号，最少2个字符"
      },
      "limit": {
        "type": "integer",
        "description": "返回结果数量上限",
        "minimum": 1,
        "maximum": 100,
        "default": 20
      },
      "status": {
        "type": "string",
        "enum": ["active", "inactive", "pending"],
        "description": "客户状态筛选"
      }
    },
    "required": ["query"]
  }
}
```

Schema 中每个关键字的作用：

| 关键字 | 作用 | AI 如何使用 |
|---|---|---|
| `type` | 字段数据类型 | 确保生成正确格式的参数值 |
| `description` | 语义解释 | 理解何时用、填什么值 |
| `required` | 必填字段列表 | 调用时确保必填字段有值 |
| `enum` | 枚举值列表 | 只从列表中选择，不自由发挥 |
| `minimum/maximum` | 数值范围 | 生成的数值在合法范围内 |
| `default` | 默认值 | 用户未提供时使用此值 |

**一句话总结**：AI 读 `description` 知道"要不要调这个 Tool"，读 `Schema` 知道"参数怎么填"。两者缺一不可。
