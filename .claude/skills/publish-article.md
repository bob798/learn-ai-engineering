---
name: publish-article
description: 技术系列文章全流程发布体系——写作规范、质量检查、多平台适配、发布工具。用于《不可逆的Agent》系列及同类技术博客。
---

# 技术文章发布体系

## 一、写作规范

### 1.1 文章长度

**公众号正文：1200-1800 字**（不含参考文献）

| 规则 | 依据 |
|------|------|
| 下限 1200 字 | 低于此信息密度不足，读者觉得"没干货" |
| 上限 1800 字 | 超过此完读率显著下降（公众号最佳完读区间 800-2000 字） |
| 每 300 字必须有一个价值点 | 数据/结论/转折/场景切换 |
| 前 150 字必须进入正题 | 铺垫超 150 字算法开始扣分 |
| 参考文献不计入正文字数 | 精简到 4-6 条，小号灰色字体 |

数据来源：
- 2026 年公众号算法权重：完读率 35% > 分享率 30% > 收藏率 20% > 互动 15%
- 完读率权重是打开率的 1.8 倍，完读率 75% 的文章比只读标题的多获得 40% 曝光
- 账号垂直度 >70% 可获 300% 曝光提升，搜一搜已占 40% 新增关注

### 1.2 文章结构

```
开头（≤150字）：场景/问题/反常识引入，直接切正题
  ↓
黑盒打开（核心论点展开）：2-4 个章节，每章 300-500 字
  ↓
伏笔（物理世界连接）：1 段，点到为止
  ↓
留一个问题（互动收束）：1 段
  ↓
系列导航 + 参考文献
```

### 1.3 写作风格

- **口语化但不随意**：像在跟一个技术同行聊天，不像在写论文
- **一篇只解决一个问题**：系列文章的优势是每篇聚焦，不要贪多
- **用具体场景代替抽象概念**：先讲灯光 Agent 翻车，再引出 ReAct
- **伏笔串联系列**：每篇末尾埋一个钩子指向后续文章
- **不加 emoji**：技术文章的严肃感靠内容不靠装饰
- **粗体用于金句和关键判断**，不用于普通强调

### 1.4 信息密度检查

每 300 字区间必须包含以下至少一项：
- 一个数据/事实（"失败率 41%-87%"）
- 一个转折/反常识（"这根本不是 AI 问题"）
- 一个具体场景（"第 5 轮又查了一遍客厅"）
- 一个可操作的判断（"默认编排，慎用协同"）

连续 300 字只有铺垫/解释而没有新信息 → 必须删减。

## 二、发布前质量检查清单

### 2.1 内容检查
- [ ] 字数在 1200-1800 范围内（`wc -m` 验证）
- [ ] 前 150 字已进入正题
- [ ] 每 300 字有一个价值点
- [ ] 没有连续 2 段纯铺垫
- [ ] 伏笔指向了系列后续文章
- [ ] "留一个问题"是开放性的、值得想的
- [ ] 参考文献精简到 4-6 条

### 2.2 系列一致性
- [ ] frontmatter 完整（title/description/abstract/status/date/series_index/tags/prev/next）
- [ ] 系列引言一致："本系列回答一个问题：当 Agent 的动作有真实世界后果时，该怎么设计。"
- [ ] 上下篇导航正确
- [ ] 公众号名统一为「我是Bob大叔」

### 2.3 视觉检查
- [ ] 封面配色统一（深紫底 + 珊瑚橙强调）
- [ ] 正文内联样式完整（微信不支持 class）
- [ ] 代码块、金句卡片、黄色提示框样式正确
- [ ] 发布服务器启动后浏览器预览无异常

## 三、多平台适配

### 3.1 公众号（主阵地）

完整文章 + 横版封面。通过 wechat HTML 的"复制正文"或 WechatSync 发布。

### 3.2 X 推文 — "观点弹药"

一个可被转发的判断，不是文章摘要。

| 规则 | 说明 |
|------|------|
| 开头 | 1.5 秒定生死：数字/反常识/转折 |
| 正文 | 1 个核心观点 + 1 个论据，≤4 行 |
| 链接 | 不放（X 算法降权），引流写在文字里 |
| hashtag | ≤2 个（2026 年 X 靠语义理解，hashtag 影响极小） |
| 总长 | ≤200 字 |

```
{反常识开头，一句话}

{1个支撑论据/数据，2-3行}

{一句金句收束}

系列第 {N} 篇 → 公众号「我是Bob大叔」
```

### 3.3 小红书 — "钩子"

引起好奇→引流完整版，不是文章缩写。

**发布界面三个字段（分别复制）：**

**标题**（≤20 字，`copyXHSTitle()`）
- 数据/反常识钩子，绝不复述文章标题
- 数字优先，或反直觉判断

**正文**（150-250 字，`copyXHSBody()`）
- 结构：钩子→痛点共鸣→1 个意外→悬念→引流
- 不放 hashtag（通过界面 # 按钮添加）
- 固定结尾：`完整版 → 公众号「我是Bob大叔」第{N}篇`

**话题**（5 个，`copyXHSTags()`）
- 固定 3 个：`#AI` `#AIAgent` `#人工智能`
- 文章相关 2 个

**轮播卡片**（3 张，3:4 竖版 1080×1440，`downloadCarousel()`）
- 第 1 张（钩子）：数据/冲突 + 原因
- 第 2 张（核心）：核心论点 + 关键数据 + 解法概要
- 第 3 张（引流）：悬念 + 公众号 CTA

### 3.4 知乎/掘金/CSDN（多平台同步）

完整文章 + 引流尾部（crossPlatformCTA）。通过 WechatSync "多平台同步"按钮发布。

## 四、发布工具 HTML 结构

### 4.1 文件命名
- `{序号}-{短名}.md` — 带 frontmatter 的正式 markdown
- `{序号}-{短名}-wechat.html` — 发布工具页面

### 4.2 视觉规范（《不可逆的Agent》系列）
```
横版封面底色: linear-gradient(135deg, #1a0a2e, #2d1b4e, #0f2027)
竖版封面底色: linear-gradient(160deg, #0a0a1a 0%, #1a0a2e 30%, #2d1050 60%, #1a0a2e 100%)
光晕: radial-gradient(ellipse at 30% 50%, rgba(120,80,200,0.15)...) + rgba(0,180,220,0.1)
标题强调色: linear-gradient(90deg, #ff6b6b, #ffa36b) — 珊瑚橙渐变
XHS标题: linear-gradient(90deg, #ff6b6b, #ffa36b, #ffdb6b)
高亮元素: rgba(255,107,107,0.15) bg / rgba(255,107,107,0.5) border / #ff6b6b text
```

### 4.3 工具栏按钮
```
素材区 | ⬇️横版封面 | ⬇️竖版封面 | 📕轮播卡片×3
📋复制正文 | 𝕏复制推文 | 📕标题 | 📕正文 | 📕话题
📤公众号同步 | 📤多平台同步
```

### 4.4 正文内联样式速查
```css
段落: font-size:16px;color:#333;margin-bottom:16px;line-height:2;text-align:justify;
h2: font-size:20px;font-weight:700;color:#1a1a1a;margin:32px 0 16px;padding-bottom:8px;border-bottom:2px solid #333;
strong: color:#1a1a1a;
code: background:#f5f5f5;padding:2px 6px;border-radius:3px;font-size:14px;color:#c7254e;border:1px solid #e8e8e8;
pre/code块: background:#1e1e1e;color:#d4d4d4;padding:20px;border-radius:6px;font-size:14px;line-height:1.7;
金句卡片: font-size:18px;font-weight:700;text-align:center;padding:20px;background:linear-gradient(135deg,#f5f0ff,#f0f7ff);border-radius:8px;border:1px solid #e0d8f0;
黄色提示: background:#fff8e1;border-left:4px solid #ffb300;padding:16px 20px;font-size:15px;color:#5d4037;
灰色引言: background:#f7f7f7;border-left:4px solid #ddd;padding:12px 16px;font-size:14px;color:#888;
分隔符: · · · (center/gray/letter-spacing:12px)
互动区: background:#f9f9f9;border-radius:8px;padding:24px;font-weight:600;
参考文献: font-size:13px;color:#666;（标题 font-size:16px）
```

## 五、发布流程

```
1. 分析文章 → 提取标题/系列编号/核心观点
2. 质量检查 → 跑 2.1-2.3 清单
3. 创建/更新 markdown → 加 frontmatter
4. 创建 wechat HTML → 封面+正文+轮播卡片+文案
5. 启动发布服务器 → 浏览器预览确认
6. 提交 GitHub → feat/fix/style 消息格式
7. 发布到各平台 → 公众号→小红书→X→多平台同步
```

### 发布服务器
```bash
kill $(lsof -t -i :8787) 2>/dev/null; sleep 1
node scripts/publish-server.mjs content/02-agent/irreversible-agent
# → http://localhost:8787/{文件名}-wechat.html
```

### Git 提交
```bash
git add "{md}" "{html}"
git commit -m "feat: 《不可逆的Agent》系列第{N}篇 — {短标题}"
git push
```
消息格式：`feat:` 新文章，`fix:` 内容优化，`style:` 样式调整

## 六、数据来源

- [2026年公众号推荐算法的5个核心指标](https://zhuanlan.zhihu.com/p/2050217665520834537) — 完读率 35%/分享率 30%/收藏率 20%/互动 15%
- [微信公众号运营2026](https://asia.marketingtochina.com/2026/06/26/wechat-official-account-growth-2026/) — 完读率权重 1.8 倍于打开率
- [2026公众号算法大变天](https://www.askuzu.com/blog/post/484.html) — 铺垫超 150 字扣分，每 300 字一个价值点
- [2026年公众号运营策略](https://m.thepaper.cn/newsDetail_forward_32292052) — 800-1500 字黄金区间
- [Elementor: Blog Post Length 2026](https://elementor.com/blog/blog-post-length/) — 技术内容 1800-2000 词
- [X Algorithm 2026](https://www.teract.ai/resources/twitter-algorithm-2026/) — 回复和收藏权重最高，hashtag 影响极小
