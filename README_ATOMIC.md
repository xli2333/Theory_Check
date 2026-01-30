# 商业案例理论核查系统 (Case Theory Verification System)

> **版本**: 2.0.0
> **状态**: 活跃开发中
> **文档日期**: 2025-12-09

## 1. 产品概述 (Product Overview)

本系统是一款专为商业案例分析设计的**智能化辅助核查平台**。它致力于解决学术或商业案例撰写过程中“理论创新性”与“知识点独特性”难以量化验证的痛点。

通过构建基于历史数据的**本地知识库**，系统利用**大语言模型 (LLM)** 的语义理解能力，对新上传的案例文件进行深度解析，并与历史库中的**理论框架 (Theories)** 和**实务知识点 (Knowledge Points)** 进行双轨制比对。

### 核心价值
*   **量化创新**: 并非简单的查重，而是通过语义分析判断“旧瓶装新酒”或“概念混淆”。
*   **分级预警**: 提供“高度重合”、“次重合”、“一般关联”三级风险提示，避免一刀切。
*   **全景溯源**: 每一处判定均可追溯至具体的历史年份、文件来源及原文语境。

---

## 2. 核心设计理念 (Core Design Philosophy)

本系统的设计遵循以下三大原则，确保工具的专业性与可用性：

### 2.1 宁滥勿缺 (High Recall Strategy)
在查重场景下，**漏报 (False Negative)** 的代价远高于**误报 (False Positive)**。
*   **提取层**: 采用宽松的提取策略，捕捉所有显性及隐性的理论提及。
*   **匹配层**: 即使是弱相关（如“长尾效应”与“长尾理论”），系统也会将其纳入候选队列，交由人工最终裁决。

### 2.2 赛道隔离 (Track Isolation)
为了避免逻辑混淆，系统实现了**双轨制独立验证**：
*   **理论赛道 (Theory Track)**: 仅比对波特五力、SWOT、PEST 等方法论框架。
*   **知识点赛道 (Knowledge Point Track)**: 仅比对“私域流量”、“数字化转型”等实务概念。
*   **优势**: 防止将“具体的营销手段”误判为“宏观的管理理论”。

### 2.3 无边框设计 (Borderless UI)
前端界面采用 **"Avant-garde Finance"** 风格：
*   抛弃传统的边框分割，依靠**留白 (Whitespace)** 与**排版 (Typography)** 构建层级。
*   色彩体系采用“深岩灰”配合“警示红/琥珀黄/参考蓝”，营造沉浸式的高级分析体验。

---

## 3. 产品功能详解 (Detailed Features)

### 3.1 智能提取引擎 (Extraction Engine)
*   **多模态解析**: 支持 PDF 文档的文本层提取与清洗。
*   **上下文感知**: 在提取关键词的同时，自动截取前后 300 字的**原文语境 (Context)**，用于后续的语义消歧。
*   **双重分类**: 自动将提取内容标记为 `category='理论'` 或 `category='知识点'`。

### 3.2 三级语义匹配 (Three-Level Semantic Matching)
AI 裁判通过对比“新词”与“库中词”，输出结构化的匹配等级：

| 等级 | 标识色 | 定义 | 示例 | 处理建议 |
| :--- | :--- | :--- | :--- | :--- |
| **高度重合 (High)** | 🔴 玫瑰红 | 同义词、翻译差异、缩写互换 | "KOL" vs "关键意见领袖" | 必须修改或删除 |
| **次重合 (Medium)** | 🟠 琥珀黄 | 包含关系、强依赖、应用变体 | "私域运营" vs "用户留存" | 需调整表述或补充差异 |
| **一般重合 (Low)** | 🔵 科技蓝 | 同一大领域、弱关联 | "短视频" vs "直播带货" | 可适度保留并规范引用 |

### 3.3 交互式仪表盘 (Interactive Dashboard)
*   **实时流式反馈**: 摒弃传统的旋转 Loading，使用文字流展示后台处理进度（如“正在构建语义空间...”、“连接历史库...”）。
*   **证据链可视化**: 点击任意查重条目，侧边栏滑出详细对比卡片，展示新旧文对比、年份分布及来源文件。

### 3.4 报告生成系统 (Reporting System)
*   **自包含 HTML**: 生成单文件 HTML 报告，内联所有 CSS/JS，无需网络即可在任意设备查看，保留交互功能。
*   **打印优化**: 针对 A4 纸张优化的 CSS `@media print` 样式，支持直接导出为 PDF 存档。

---

## 4. 技术栈 (Tech Stack)

### 4.1 后端 (The Brain)
*   **Runtime**: Python 3.9+
*   **Web Framework**: **FastAPI** (异步高性能 API 服务)
*   **Database**: **SQLite** (轻量级、无需配置的本地知识库存储)
*   **AI Integration**: 
    *   **LLM SDK**: 集成大语言模型 API 进行推理。
    *   **Prompt Engineering**: 精心设计的结构化提示词（Few-shot learning）。
*   **PDF Processing**: `pdfplumber` (高精度文本提取)
*   **Utilities**: `pydantic` (数据校验), `python-multipart` (文件上传)

### 4.2 前端 (The Face)
*   **Core**: **React 18** + **TypeScript**
*   **Build Tool**: **Vite** (极速构建)
*   **Styling**: **Tailwind CSS** (原子化 CSS 引擎)
*   **Motion**: **Framer Motion** (平滑的转场与加载动画)
*   **Icons**: `lucide-react`
*   **Visualization**: `recharts` (数据图表)

---

## 5. 目录结构 (Directory Structure)

```text
Project_Root/
├── knowledge_base.db           # 核心数据库（存储历史案例特征）
├── server.py                   # 后端主程序 (FastAPI)
├── build_knowledge_base_v2.py  # 知识库构建脚本
├── Logic_Design.md             # 核心逻辑设计文档
├── THREE_LEVEL_MATCHING.md     # 匹配算法详解
├── requirements.txt            # Python 依赖清单
└── frontend/                   # 前端工程目录
    ├── src/
    │   ├── components/         # UI 组件 (UploadZone, ResultDashboard...)
    │   ├── App.tsx             # 主应用入口
    │   └── main.tsx            # 渲染入口
    ├── tailwind.config.js      # 样式配置
    └── vite.config.ts          # 构建配置
```

---

## 6. 安装与部署 (Installation & Setup)

### 前置要求
*   Python 3.9 或更高版本
*   Node.js 18.0 或更高版本
*   有效的 LLM API 密钥 (需在环境变量中配置)

### 步骤 1: 后端环境配置

1.  **创建虚拟环境**:
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

2.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **配置环境变量**:
    在根目录创建 `.env` 文件（或直接设置系统环境变量），填入 API Key：
    ```env
    API_KEY=your_api_key_here
    ```

4.  **初始化知识库** (如果尚未存在):
    ```bash
    # 扫描指定目录下的历史 PDF 并入库
    python build_knowledge_base_v2.py
    ```

5.  **启动服务**:
    ```bash
    python server.py
    # 服务将运行在 http://localhost:8000
    ```

### 步骤 2: 前端环境配置

1.  **进入前端目录**:
    ```bash
    cd frontend
    ```

2.  **安装依赖**:
    ```bash
    npm install
    ```

3.  **启动开发服务器**:
    ```bash
    npm run dev
    # 前端将运行在 http://localhost:5173
    ```

---

## 7. 使用流程 (Workflow)

1.  **准备文件**: 将需要核查的案例导出为 PDF 格式。
2.  **上传分析**: 打开前端页面，拖拽 PDF 至 "Upload Zone"。
3.  **等待解析**: 观察屏幕上的实时处理日志，系统将并行执行提取与匹配任务。
4.  **查看报告**:
    *   **概览**: 查看顶部的综合创新指数与风险分布。
    *   **详情**: 点击列表项，检查具体的理论冲突证据。
5.  **导出存档**: 点击右上角 "Export Report"，选择保存为 HTML 或打印为 PDF。

---

## 8. 注意事项 (Notes)

*   **编码格式**: 项目统一使用 **UTF-8** 编码，处理中文字符串时需特别注意。
*   **隐私安全**: 请确保上传的文件不包含敏感的个人身份信息 (PII)，尽管系统主要在本地运行，但摘要提取依赖云端 API。
*   **网络连接**: 运行期间需要稳定的互联网连接以访问 LLM 推理服务。

---

*Powered by Advanced Semantic Analysis Technology*
