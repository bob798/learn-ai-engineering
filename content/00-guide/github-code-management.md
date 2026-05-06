# GitHub 代码管理规范

> 关联 Issue: [#3](https://github.com/bob798/learn-ai-engineering/issues/3)

## 1. 分支策略

### GitHub Flow（推荐轻量级项目）

```
main（始终可部署）
 └── feature/xxx → PR → merge → 删除分支
```

规则：
- `main` 分支永远保持稳定可部署
- 所有开发在功能分支进行
- 通过 PR 合并，合并后删除分支

### Git Flow（适合有发布周期的项目）

```
main ─────────────────────────── 生产环境
  └── develop ──────────────── 开发主线
        ├── feature/xxx ────── 功能开发
        ├── release/1.0 ────── 发布准备
        └── hotfix/xxx ─────── 紧急修复
```

### 分支命名规范

| 前缀 | 用途 | 示例 |
|------|------|------|
| `feature/` | 新功能 | `feature/full-text-search` |
| `fix/` | 修复 Bug | `fix/link-404` |
| `docs/` | 文档 | `docs/rag-guide` |
| `refactor/` | 重构 | `refactor/api-layer` |
| `chore/` | 工具/配置 | `chore/ci-setup` |

---

## 2. Commit 规范（Conventional Commits）

### 格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type 类型

| Type | 说明 | 示例 |
|------|------|------|
| `feat` | 新功能 | `feat: 全站搜索功能（Cmd+K）` |
| `fix` | 修复 | `fix: 修复内部链接 404` |
| `docs` | 文档 | `docs: 添加 RAG 论文速查表` |
| `refactor` | 重构 | `refactor: 抽取公共组件` |
| `style` | 格式 | `style: 统一缩进为 2 空格` |
| `test` | 测试 | `test: 补充搜索模块单测` |
| `chore` | 构建/工具 | `chore: 升级 Next.js 版本` |
| `perf` | 性能 | `perf: 图片懒加载优化` |

### 规则

- subject 不超过 50 字符
- body 说明 **为什么** 改，而非改了什么（diff 已经说明了）
- footer 关联 issue：`closes #3`

### 参考规范

- [Conventional Commits 1.0.0](https://www.conventionalcommits.org/)
- [Angular Contributing Guide](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit)

---

## 3. Issue 管理

### 标签体系

建议按三个维度设计标签：

**类型（Type）**
- `bug` — 缺陷报告
- `feature` — 新功能请求
- `docs` — 文档相关
- `question` — 问题讨论
- `enhancement` — 改进优化

**优先级（Priority）**
- `P0-critical` — 阻塞性问题，立即处理
- `P1-high` — 重要，当前迭代完成
- `P2-medium` — 一般，排期处理
- `P3-low` — 低优，有空再做

**状态（Status）**
- `needs-triage` — 待分类
- `confirmed` — 已确认
- `in-progress` — 进行中
- `wontfix` — 不予修复
- `duplicate` — 重复

### Issue 模板

在 `.github/ISSUE_TEMPLATE/` 下创建模板：

```yaml
# .github/ISSUE_TEMPLATE/config.yml
blank_issues_enabled: true
contact_links:
  - name: 讨论区
    url: https://github.com/xxx/discussions
    about: 一般性问题请到讨论区
```

```markdown
<!-- .github/ISSUE_TEMPLATE/bug_report.md -->
---
name: Bug 报告
about: 报告一个问题
labels: bug
---

## 问题描述

## 复现步骤
1.
2.
3.

## 期望行为

## 实际行为

## 环境信息
- OS:
- 浏览器:
- Node 版本:
```

### Milestone（里程碑）

用 Milestone 将 issue 按阶段归组：

```
v0.1 基础框架搭建
v0.2 RAG 系列内容
v0.3 Agent 系列内容
```

---

## 4. Pull Request 流程

### PR 模板

创建 `.github/pull_request_template.md`：

```markdown
## 概要
<!-- 这个 PR 做了什么 -->

## 变更类型
- [ ] 新功能
- [ ] Bug 修复
- [ ] 文档更新
- [ ] 重构
- [ ] 其他

## 关联 Issue
closes #

## 测试说明
<!-- 如何验证这个变更 -->

## 截图（如有 UI 变更）
```

### PR 最佳实践

1. **小而专注** — 一个 PR 只解决一个问题
2. **标题清晰** — 遵循 commit 规范（`feat:`, `fix:` 等）
3. **描述完整** — 写明 why 和 how
4. **关联 Issue** — 使用 `closes #N` 自动关闭
5. **CI 通过** — 合并前确保所有检查绿色
6. **及时 Review** — 不超过 24 小时

### 合并策略

| 策略 | 适用场景 |
|------|----------|
| Merge commit | 保留完整历史，适合大功能 |
| Squash merge | 压缩为一个 commit，保持主线整洁 |
| Rebase merge | 线性历史，适合小改动 |

---

## 5. 自动化（GitHub Actions）

### 常用自动化场景

```yaml
# .github/workflows/ci.yml
name: CI
on:
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npm run lint

  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npm run build
```

### 推荐 Bot

| Bot | 功能 |
|-----|------|
| **stale** | 自动关闭长期无活动 issue |
| **dependabot** | 自动更新依赖 |
| **codecov** | 代码覆盖率报告 |
| **semantic-release** | 自动版本发布 |

---

## 6. 完整工作流示例

```bash
# 1. 从 issue 出发
gh issue create --title "feat: 添加暗色模式" --label "feature"
# Created issue #10

# 2. 创建分支
git checkout -b feature/dark-mode

# 3. 开发 + 提交
git add .
git commit -m "feat: 添加暗色模式支持

实现 CSS 变量切换方案，支持系统偏好检测

closes #10"

# 4. 推送 + 创建 PR
git push -u origin feature/dark-mode
gh pr create --title "feat: 添加暗色模式" --body "closes #10"

# 5. Review → CI 通过 → 合并 → Issue 自动关闭
```

---

## 参考资源

- [GitHub Flow 官方文档](https://docs.github.com/en/get-started/using-github/github-flow)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Angular Contributing Guide](https://github.com/angular/angular/blob/main/CONTRIBUTING.md)
- [Kubernetes 社区贡献指南](https://github.com/kubernetes/community/tree/master/contributors/guide)
- [开源指南 - 最佳实践](https://opensource.guide/)
