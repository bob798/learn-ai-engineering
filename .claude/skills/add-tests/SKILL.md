---
name: add-tests
description: 给已有代码补充单元测试。先读实现，识别可测函数，逐函数补充测试用例，确保全部通过。
disable-model-invocation: true
allowed-tools: Read, Edit, Write, Bash, Grep, Glob
---

# 给已有代码补充单元测试

目标：$ARGUMENTS（文件路径或模块名，留空则分析当前项目）

---

## Step 1：读代码，识别可测单元

读取目标文件，列出所有函数/方法，按可测性分类：

| 类型 | 特征 | 测试优先级 |
|---|---|---|
| 纯逻辑函数 | 输入→输出，无副作用 | **P0，必测** |
| 有异常处理的函数 | try/except，错误路径 | **P0，必测** |
| 依赖外部 IO 的函数 | API 调用、文件读写、DB | P1，mock 后测 |
| 主流程入口 | main()、交互循环 | P2，集成测试 |

**只测 P0 和 P1，跳过 P2**（主流程靠手动验证）。

## Step 2：检查已有测试，避免重复

```bash
find . -name "test_*.py" -o -name "*_test.py" | head -20
```

读取已有测试文件，了解：
- 已覆盖哪些函数
- Mock 策略（避免风格不一致）
- 测试框架（pytest / unittest）

## Step 3：逐函数写测试

每个 P0/P1 函数，至少覆盖：

1. **Happy path**：正常输入，验证输出
2. **Edge cases**：空输入、边界值、特殊字符
3. **Error handling**：异常输入、API 失败、文件不存在

写法原则：
- 每个 test 只验证一件事
- 测试名说明场景：`test_xxx_when_yyy_returns_zzz`
- Mock 只 mock 必要的外部依赖，不 mock 被测函数本身
- 断言具体值，不断言 `is not None`

## Step 4：运行测试，修复失败

```bash
python3 -m pytest <测试文件> -v
```

若测试失败：
- 先判断是**测试写错了**还是**代码有 bug**
- 如果是代码 bug，标注出来，不要悄悄改代码
- 如果是测试写错了，修正测试逻辑

## Step 5：输出覆盖报告

```
新增测试：X 个
覆盖函数：[函数列表]
跳过函数：[函数名] — 原因（主流程/太依赖外部）
发现潜在 bug：[如有]
运行结果：X passed, 0 failed
```
