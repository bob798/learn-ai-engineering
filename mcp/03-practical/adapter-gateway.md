# 异构系统接入：Adapter 模式与 Gateway 设计

现实世界的企业系统永远是异构的——有只提供 REST API 的老系统，有用 gRPC 的内部服务，有根本没有任何接口的遗留系统。MCP 的应对策略是**适配器模式**：在 MCP Server 这一层做协议转换，上层 AI 不感知差异。

---

## 情况一：系统只有 REST API

### 方案：写一个 MCP Adapter Server

```
AI Client ─── MCP协议 ───→ [MCP Adapter Server]
                                    │
                        内部转换为 REST API 调用
                                    │
                              老系统 REST API
```

Adapter Server 就是一个普通的 MCP Server，Tool 的实现内部调用老系统的 REST API。AI 完全不知道背后是什么，只看到 MCP 接口。

```python
@mcp.tool()
def get_customer(customer_id: str) -> dict:
    """获取客户详情，已知 customer_id 时使用"""
    # 内部调 REST API，AI 不感知
    response = requests.get(
        f"http://crm-service/api/customers/{customer_id}",
        headers={"Authorization": f"Bearer {os.environ['CRM_TOKEN']}"}
    )
    return response.json()
```

**对老系统零侵入**——不需要修改老系统代码，只需要在外层加一个 Adapter。

### 有没有自动生成的方案？

**有。只要系统有 OpenAPI/Swagger 文档，可以接近零代码自动生成 MCP Server：**

```bash
npm install -g openapi-mcp

# 指向任何 OpenAPI spec 文件，自动生成 MCP Server
openapi-mcp --spec https://api.stripe.com/openapi.json \
            --base-url https://api.stripe.com \
            --api-key $STRIPE_KEY
```

工具从 OpenAPI spec 中读取接口定义，自动生成对应的 MCP Tools，Tool description 从 OpenAPI 注释自动提取。

**优先级建议**：
1. 查有没有官方 MCP Server（GitHub/Slack/PostgreSQL 等已有官方实现）
2. 查有没有 OpenAPI spec（可以自动生成）
3. 最后才手写 Adapter

---

## 情况二：混合 gRPC + REST + 原生 MCP

### 方案：MCP Gateway（统一网关）

```
AI Client ─── MCP协议 ─→ [MCP Gateway]
                               │
           ┌───────────┬───────┴──────────┐
           ↓           ↓                  ↓
      原生MCP       gRPC服务          REST API
       Server        ↓                   ↓
     (直接路由)  [gRPC→MCP适配层]  [REST→MCP适配层]
```

### Gateway 的核心代码结构

```python
from mcp.server import FastMCP

mcp = FastMCP("enterprise-gateway")

# 包装内部 REST API
@mcp.tool()
def get_customer(customer_id: str) -> dict:
    """获取客户基本信息"""
    return crm_rest_client.get(f"/customers/{customer_id}")

# 包装内部 gRPC 服务
@mcp.tool()
def get_orders(customer_id: str, limit: int = 20) -> list:
    """获取客户订单列表"""
    return order_grpc_stub.ListOrders(
        customer_id=customer_id, limit=limit
    )

# Gateway 核心价值：聚合多个服务
# AI 调用 1 次，背后是 3 个服务的并发请求
@mcp.tool()
def get_full_profile(customer_id: str) -> dict:
    """
    获取客户完整画像，含基本信息、订单记录和风控评分。
    [何时用] 需要全面了解客户情况时（如处理投诉、销售跟进）
    [注意] 调用耗时较长（~500ms），简单查询用 get_customer 更快
    """
    import asyncio
    customer, orders, risk = asyncio.gather(
        crm_rest_client.async_get(f"/customers/{customer_id}"),
        order_grpc_stub.async_list(customer_id=customer_id),
        risk_grpc_stub.async_score(customer_id=customer_id)
    )
    return {
        "customer": customer,
        "orders": orders,
        "risk_score": risk["score"]
    }
```

### Gateway 的完整职责

Gateway 不只是协议转换，还承担以下责任：

```
AI Client
    │ MCP协议
    ▼
┌─────────────────────────────┐
│         MCP Gateway         │
│  ┌────────────────────────┐ │
│  │ 权限控制               │ │ ← 这个 AI 用户能用哪些 Tool
│  │ 限流 / 配额            │ │ ← 防止 AI 无限调用消耗资源
│  │ 审计日志               │ │ ← 记录 AI 做了什么，可追溯
│  │ Tool 编排（聚合多服务）│ │ ← 一个 Tool = 多个服务调用
│  │ 错误处理 & 重试        │ │ ← 统一的错误处理策略
│  └────────────────────────┘ │
└─────────────────────────────┘
```

### 重要认知澄清

**Gateway 不是把所有内部服务全部暴露为 MCP**。

错误理解：把 20 个内部服务的所有接口都包装成 MCP Tool → AI 看到几百个 Tool，选择混乱，性能下降。

正确理解：**选择性暴露 AI 需要用到的能力**，10 个内部服务可能只暴露 15 个精心设计的 Tool，每个 Tool 都有清晰的 description 和明确的使用场景。

---

## 各类系统的接入策略

| 系统类型 | 推荐方案 | 开发成本 | 备注 |
|---|---|---|---|
| 有 OpenAPI spec 的 REST API | openapi-mcp 自动生成 | 极低（1天内） | 最优先尝试 |
| 无 spec 的 REST API | 手写 MCP Adapter | 低（2-3天） | 对老系统零侵入 |
| gRPC 服务（有 proto 文件） | 手写 Adapter，内部用 gRPC stub | 中（2-3天） | 需要 proto 文件 |
| 消息队列（Kafka/MQ） | Resource 暴露消费接口，Tool 发布消息 | 中 | 需要设计 schema |
| 数据库直连 | MCP Server 直接操作 DB | 低，但**风险极高** | 必须严格权限控制 |
| 完全封闭系统（无接口） | RPA/屏幕抓取 + MCP 包装 | 高，且脆弱 | 最后手段 |

---

## 开发成本现实估算

| 规模 | 内容 | 工时 |
|---|---|---|
| 单服务 Adapter | 包装一个 REST/gRPC 服务的核心接口 | 1-3天 |
| 完整 Gateway | 多服务聚合 + 权限 + 限流 + 审计 | 1-2周 |
| 生产级稳定性 | 错误处理、重试、监控、文档 | 1个月+ |

**没有银弹**：MCP 标准化了"接口协议"，没有标准化"业务逻辑翻译"。把系统的 API 翻译成 AI 友好的 Tool，这个业务理解的工作永远需要人来做。
