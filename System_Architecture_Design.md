# FDC 商业案例查重系统 - 全栈架构设计方案 (Web Platform Design)

**状态**: 待确认
**日期**: 2025-11-27
**设计目标**: 构建一个**前卫、无边框、高级金融风格**的 Web 应用，用于商业案例的 AI 语义查重与报告生成。

---

## 1. 整体技术栈 (Tech Stack)

为了实现“高级交互”与“专业报告”的结合，我们采用前后端分离架构：

*   **前端 (The Face)**:
    *   **框架**: React 18 (Vite)
    *   **样式引擎**: Tailwind CSS (核心)
    *   **UI 风格**: "Borderless Financial" (无边框、大留白、衬线/无衬线混排、深色模式或高冷灰白)
    *   **交互动画**: Framer Motion (平滑的转场、加载态、数据可视化展开)
    *   **组件库**: ShadcnUI (Headless 组件，允许我们完全自定义为“无边框”风格)
*   **后端 (The Brain)**:
    *   **框架**: FastAPI (Python) - 高性能，适合处理 AI 异步请求。
    *   **核心逻辑**: 也就是我们之前写好的 `check_duplication_final.py` 的逻辑封装成 API。
    *   **数据库**: SQLite (`knowledge_base.db`) - 继续复用已建好的库。
    *   **AI 引擎**: Google Gemini API via Proxy。

---

## 2. 视觉设计语言： "Avant-garde Finance"

我们抛弃传统的“后台管理系统”那种方框套方框的设计。

*   **无边框 (Borderless)**: 内容的区分不靠 `border`，而靠**间距 (Spacing)**、**字号 (Typography)** 和微妙的**背景色差 (Surface Tones)**。
*   **字体 (Typography)**:
    *   *标题*: **Playfair Display** 或 **Noto Serif** (体现金融的厚重与历史感)。
    *   *正文*: **Inter** 或 **Roboto** (体现数据的冷静与精准)。
    *   *数字*: **JetBrains Mono** (等宽字体，体现专业分析感)。
*   **色彩 (Palette)**:
    *   *主色*: **Deep Slate** (`slate-900`) 或 **Midnight Blue**。
    *   *强调色*: **Electric Gold** (金色，象征价值) 或 **International Orange** (极简的点缀，用于高风险警告)。
    *   *背景*: 灰白色系 (`zinc-50`)，像高级信纸的质感。

---

## 3. 用户体验流程 (UX Flow)

### 3.1 首页 / 仪表盘 (The Landing)
*   **布局**: 极简。屏幕中央只有一个极其精致的 **"Upload Zone" (上传区)**。
*   **交互**:
    *   支持拖拽 PDF。
    *   或者切换 Tab 输入一段纯文本（自定义理论）。
*   **视觉**: 玻璃拟态或纯粹的留白，没有任何多余的线条。

### 3.2 分析过程 (The Processing)
*   不要使用普通的旋转 Loading 图标。
*   **设计**: 使用**文字流**或**进度波形**。
    *   屏幕上依次浮现：“正在解析语义架构...”、“连接 FDC 历史库...”、“AI 裁判正在裁决...”。
    *   给用户一种“正在进行深度计算”的高级感。

### 3.3 结果呈现 (The Insight Dashboard)
分析完成后，页面平滑过渡到结果页。

*   **顶部**: 关键指标（创新度、风险指数），用超大字号展示。
*   **主体**: **分栏式布局**。
    *   *左侧*: 理论框架风险。
    *   *右侧*: 实务知识点风险。
*   **列表交互**:
    *   每一行查重记录平时是一个简洁的摘要。
    *   **点击展开**: 优雅地向下滑出，展示“新旧文对比”和“证据链”。
    *   **高亮**: 风险词汇用背景色块高亮，而不是文字变色。

### 3.4 报告输出 (The Deliverable)
页面右上角悬浮一个 **"Export"** 按钮。
*   点击后弹出模态框：
    *   选项 A: **Interactive HTML** (保留所有交互、筛选功能，适合发给客户在浏览器看)。
    *   选项 B: **Print-Ready PDF** (自动调用浏览器的打印渲染，去除交互元素，优化为 A4 排版)。

---

## 4. 核心功能模块规划

### 4.1 知识库管理 (Knowledge Base)
虽然主要功能是查重，但在前端需要一个入口查看“底库”。
*   展示目前库里有多少文件、覆盖了哪些年份。
*   （可选）支持手动上传新文件入库。

### 4.2 查重分析器 (Analyzer)
*   **输入**: PDF 文件流。
*   **处理**: 后端调用 Gemini -> SQLite 匹配 -> 结果结构化。
*   **输出**: JSON 给前端渲染。

### 4.3 报告生成器 (Reporter)
*   **HTML 下载**: 前端将当前的渲染状态打包，或者后端直接返回一个服务端渲染（SSR）的静态 HTML 文件。
*   **设计**: 这个 HTML 文件是**自包含**的（CSS/JS 内联），不需要联网也能看，且样式与网站一致，但更适合阅读。

---

## 5. 开发步骤规划

如果您确认这个设计方案，我将分两步为您构建：

### 第一步：后端 API (Python/FastAPI)
1.  搭建 FastAPI 服务。
2.  移植之前的查重逻辑，封装为 `/api/analyze` 接口。
3.  提供 `/api/export` 接口生成下载文件。

### 第二步：前端界面 (React/Tailwind)
1.  初始化 React 项目 (Vite)。
2.  配置 Tailwind CSS 主题（字体、颜色）。
3.  开发组件：
    *   `Dropzone` (无边框上传)
    *   `RiskCard` (查重结果卡片)
    *   `ReportView` (预览/打印视图)
4.  联调接口。

---

## 6. 报告样式的预期效果 (Preview of Vibe)

> *Imagine a page that looks like a simplified Financial Times article mixed with a Stripe dashboard.*
> *Paper-like background. Sharp, black typography. No borders between sections, only generous whitespace. Risk items are highlighted with a soft red marker effect.*

请确认：**这种“无边框、重排版、金融极简风”是否是您想要的方向？**
