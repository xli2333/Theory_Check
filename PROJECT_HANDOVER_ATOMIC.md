# 原子级项目交接文档 (Project Atomic Handover Documentation)

> **状态**: [框架草案] - 等待逐步填充细节
> **目标**: 实现对项目代码、逻辑、数据的原子级无歧义描述。

## 1. 项目元定义 (Project Meta-Definition)
*   **1.1 核心目的**: 本项目服务于计算社会学（Computational Social Science）范式的硕博论文研究。
    *   **研究对象**: FDC (Future Designers Challenge) 游戏行业案例库 (2021-2025)。
    *   **核心问题**: 探究“学术理论”在“行业实践”中的映射与应用情况。
    *   **具体目标**: 自动化检测并统计行业案例中“使用”了哪些学术理论，分析“直接引用”与“隐含使用”的差异，揭示理论在实践中的演变路径。

*   **1.2 核心原子单位 (The Atomic Unit)**: 系统处理的最小不可分割单元是 **“理论条目 (Theory Entry)”**。
    *   一个完整的理论条目必须包含：
        *   **Standard Name**: 理论的标准学术名称 (Unique ID)。
        *   **Category**: 显性/隐性分类 (Type)。
        *   **Original Text**: 原文中的具体表述 (Evidence)。
        *   **Context**: 原文上下文片段 (Proof)。
        *   **Source**: 来源文件及年份 (Metadata)。

*   **1.3 方法论范式 (Methodology Paradigm)**:
    *   **Direct Theory (直接理论)**: 文本显式提及理论名称（如“根据SWOT分析...”）。这代表了从业者有意识的理论应用。
    *   **Implied Theory (隐含理论)**: 文本未提及名称，但逻辑/流程完全符合特定理论框架（如描述了“把关”过程但未提“把关人理论”）。这代表了行业实践中内化了的学术逻辑，是本项目的挖掘重点。

## 2. 数据流拓扑图 (Data Flow Topology)

```mermaid
graph TD
    subgraph Physical Layer [物理存储层]
        PDFs[FDC PDF Files<br/>(C:/Users/.../FDC-xx/*.pdf)]
    end

    subgraph Extraction Layer [提取与处理层]
        Script[build_knowledge_base_v2.py]
        LLM[Google Gemini 2.5 API]
        Prompt[Academic Expert Prompt]
        
        PDFs -->|pdfplumber Text Stream| Script
        Script -->|Raw Text + Prompt| LLM
        LLM -->|JSON Structure| Script
    end

    subgraph Persistence Layer [持久化层]
        DB[(knowledge_base_v2.db<br/>SQLite3)]
        TableFiles[Table: files]
        TableTheories[Table: theories]
        
        Script -->|Insert Metadata| TableFiles
        Script -->|Insert Theories| TableTheories
    end

    subgraph Service Layer [服务与算法层]
        API[server.py<br/>(Python Backend)]
        Stats[Statistics Module]
        Search[Search Logic]
        
        DB <-->|SQL Queries| API
    end

    subgraph Presentation Layer [展示交互层]
        UI[Frontend UI<br/>(React + Vite)]
        Dashboard[ResultDashboard.tsx]
        
        API <-->|JSON Response| UI
        UI -->|Render| Dashboard
    end
```

*   **路径节点 (Path Nodes)**:
    1.  **物理层 (Physical Source)**: 
        *   路径: `C:\Users\LXG\CaseTheoryCheck\FDC-21` 至 `FDC-25`。
        *   格式: 非结构化 PDF 文件。
    2.  **提取层 (Extraction Engine)**:
        *   核心组件: `build_knowledge_base_v2.py`。
        *   核心动作: 文本清洗 -> LLM 推理 (区分 Direct/Implied) -> JSON 序列化。
    3.  **持久层 (Persistence)**:
        *   位置: 项目根目录 `knowledge_base_v2.db`。
        *   作用: 永久存储提取结果，作为后续统计分析的唯一数据源。
    4.  **服务层 (Backend Service)**:
        *   组件: `server.py`。
        *   作用: 提供无状态 API，将数据库行 (Rows) 转换为前端对象。
    5.  **展示层 (Presentation)**:
        *   组件: `frontend/` (React App)。
        *   作用: 可视化呈现理论分布，提供人工核查界面。

## 3. 模块级原子解构 (Module-Level Atomic Breakdown)

### 3.1 提取引擎 (The Extraction Engine)
*   **核心脚本**: `build_knowledge_base_v2.py`
*   **运行模式**: CLI 批处理 (Batch Processing)。
*   **原子级执行流程 (Atomic Execution Flow)**:
    1.  **初始化 (Initialization)**:
        *   设置环境变量 `HTTP(S)_PROXY` 指向本地代理 (端口 7897)。
        *   初始化 Gemini 2.5 Flash 客户端 (`genai.configure`).
        *   连接 SQLite 数据库，若表不存在则创建 (`files`, `theories`)。
        *   **控制台输出**: `--- Knowledge Base Builder V2 (Theories Only) ---`
    2.  **文件发现 (Discovery)**:
        *   使用 `os.walk` 递归扫描 `ROOT_DIR`。
        *   **过滤器**: 仅保留路径包含 `FDC-` 或 `FDC-2` 的 `.pdf` 文件。
        *   **去重**: 读取数据库 `files` 表，构建 `processed_files` 集合，跳过已存在文件。
        *   **控制台输出**: `Found X PDF files... Skipping Y already processed...`
    3.  **处理循环 (Processing Loop)**:
        *   对于每一个未处理的 PDF:
            *   **Step A (Read)**: `pdfplumber` 提取全文。若为空，记录至 `failed_files_v2.txt` 并跳过。
            *   **Step B (Display)**: `sys.stdout` 打印进度 `Processing [i/N]: filename...` (不换行)。
            *   **Step C (AI Inference)**: 
                *   构造 Prompt (见下文逻辑核)。
                *   调用 `model.generate_content` (Timeout: 600s)。
                *   **重试机制**: 若失败 (如 Network Error)，休眠 5s 后重试，最多 3 次。
            *   **Step D (Parse & Save)**: 
                *   清洗 AI 返回的 JSON (去除 Markdown 代码块)。
                *   解析为 Python Dict。
                *   执行 SQL 事务写入 `files` 和 `theories` 表。
            *   **Step E (Feedback)**: 
                *   成功: `-> [Success] Saved N items (X.Xs)`
                *   失败: `\n[FAILED] filename after 3 attempts.`
*   **逻辑核 (Prompt Logic)**:
    *   **指令**: "Your SOLE GOAL is to extract Theories... categorize into Direct Theory (explicit name) OR Implied Theory (logic/framework)."
    *   **关键约束**: 必须提取 `standard_name` (标准化名称) 和 `context` (原文证据)。
*   **异常日志**:
    *   文件: `failed_files_v2.txt` (UTF-8 编码)。
    *   格式: `Time | Filename | Reason`。

### 3.2 数据库原子结构 (Database Schema V2)
*   **文件**: `knowledge_base_v2.db` (SQLite3)
*   **Table: `files` (文件索引表)**:
    *   `id` (INTEGER PK): 唯一自增 ID。
    *   `filename` (TEXT UNIQUE): 文件名 (如 `FDC-2021-001.pdf`)，用于去重。
    *   `filepath` (TEXT): 文件的完整系统路径。
    *   `title` (TEXT): 文章标题。
    *   `publish_date` (TEXT): 发布时间 (YYYY-MM-DD)。
    *   `processed_at` (TEXT): 入库时间戳。
*   **Table: `theories` (理论明细表)**:
    *   `id` (INTEGER PK): 唯一自增 ID。
    *   `file_id` (INTEGER FK): 外键，关联 `files.id`。
    *   `category` (TEXT): **核心分类字段**，枚举值为 `'Direct Theory'` 或 `'Implied Theory'`。
    *   `standard_name` (TEXT): 理论的学术标准名称 (如 "Agenda Setting Theory")，用于后续聚合统计。
    *   `original_text` (TEXT): 原文中实际出现的词汇 (如 "议程设置功能")。
    *   `context` (TEXT): 提取点前后的文本片段 (1-2句话)，作为人工核查的证据。

### 3.3 后端服务接口 (Backend API)
*   **核心脚本**: `server.py` (FastAPI / Flask)
*   **运行模式**: 常驻服务 (Daemon)。
*   **WebSocket 协议原子流程 (`/ws/{client_id}`)**:
    1.  **连接建立 (Handshake)**: 客户端连接，服务器接受 (`await websocket.accept()`)，启动心跳任务 (10s interval)。
    2.  **数据接收**: 服务器等待 `websocket.receive_bytes()` (限制 50MB)。
    3.  **阶段 1: 解析 (Parsing)**:
        *   发送消息: `{"step": "start", "progress": 10}`。
        *   动作: `extract_text_from_pdf_bytes` 提取文本。
    4.  **阶段 2: 提取 (Extraction)**:
        *   发送消息: `{"step": "extract", "progress": 30}`。
        *   动作: 调用 `extract_new_concepts` (Gemini 2.5 Pro)。
        *   **逻辑**: 区分 "Explicit" (显性) 和 "Implicit" (隐性) 理论。
    5.  **阶段 3: 数据库加载 (DB Fetch)**:
        *   发送消息: `{"step": "db_fetch", "progress": 50}`。
        *   动作: SQL 查询 `SELECT DISTINCT standard_name FROM knowledge_points` 获取全量历史标签。
    6.  **阶段 4: AI 映射 (AI Mapping)**:
        *   发送消息: `{"step": "match", "progress": 70}`。
        *   动作: 调用 `ai_strict_mapping`。
        *   **核心算法**: 将新提取的术语与历史标签进行对比，判定匹配等级 (High/Medium/Low)。
    7.  **阶段 5: 归一化合并 (Consolidation)**:
        *   发送消息: `{"step": "consolidate", "progress": 95}`。
        *   动作: 调用 `consolidate_results` (Gemini 2.0 Flash Exp)。
        *   **目的**: 将 "RBV" 和 "资源基础观" 合并为同一条目，去重统计。
    8.  **阶段 6: 完成 (Completion)**:
        *   发送消息: `{"step": "done", "progress": 100, "data": FinalReportJSON}`。
        *   **数据负载**: 包含 `summary` (风险等级、重复率) 和 `results` (带有证据链的理论列表)。
*   **HTTP 导出接口**:
    *   `POST /api/export`: 接收完整的 JSON 报告数据，生成带有 SVG 图表的 HTML 文件流 (`report_type="paper"` 或 `"dashboard"`)，触发浏览器下载。

### 3.4 前端组件状态 (Frontend Component State)
*   **目录**: `frontend/src/components`
*   **核心组件与 WebSocket 联动逻辑**:
    *   **`UploadZone.tsx`**: 
        *   **动作**: 建立 WebSocket 连接 (`ws://localhost:8000/ws/{clientId}`).
        *   **发送**: 上传 PDF 二进制数据 (`websocket.send(arrayBuffer)`).
    *   **`LoadingState.tsx` (进度同步)**: 
        *   **状态订阅**: 监听 WebSocket `onmessage`.
        *   **UI 映射**:
            *   收到 `{"step": "start"}` -> 显示 "解析文档结构 (10%)"
            *   收到 `{"step": "extract"}` -> 显示 "AI 深度提取理论 (30%)"
            *   收到 `{"step": "db_fetch"}` -> 显示 "加载历史索引 (50%)"
            *   收到 `{"step": "match"}` -> 显示 "语义三级匹配 (70%)"
            *   收到 `{"step": "consolidate"}` -> 显示 "概念归一化 (95%)"
    *   **`ResultDashboard.tsx` (渲染引擎)**: 
        *   **输入 Props**: 接收 `step: done` 返回的 `data` 对象。
        *   **渲染逻辑**:
            *   **Summary区域**: 解析 `data.summary.risk_level`，动态渲染颜色 (红/橙/绿)。
            *   **列表区域**: 遍历 `data.results.explicit_theories` 和 `implicit_theories`。
            *   **交互**: 点击理论卡片 -> 展开 `<details>` -> 展示 `context` (当前文稿语境) 和 `evidence` (历史匹配证据链)。
    *   **`PasswordGate.tsx`**: 简单的客户端路由守卫，阻止未授权访问。

## 4. 关键算法与逻辑 (Core Algorithms & Logic)
*   **4.1 三级匹配逻辑 (Three-Level Matching)**:
    *   **位置**: `server.py` -> `_map_batch` 函数 Prompt。
    *   **输入**: 新提取的术语列表 List[A], 历史数据库标签 List[B]。
    *   **判别标准 (由 AI 执行)**:
        1.  **Level 1: High (高度重合)**: 
            *   定义: 完全相同的概念，仅表述不同（同义词、中英互译、简写）。
            *   例: "SWOT" vs "SWOT分析法"; "Porter 5 Forces" vs "波特五力模型"。
            *   **业务含义**: 直接且确定的重复引用，风险最高。
        2.  **Level 2: Medium (次重合)**:
            *   定义: 同一概念的子集、具体应用或特定侧面。
            *   例: "SWOT分析" vs "内部优势分析"; "品牌定位" vs "市场定位策略"。
            *   **业务含义**: 逻辑高度相关，属于同一理论家族。
        3.  **Level 3: Low (一般重合)**:
            *   定义: 概念相关，变体或延伸，但核心定义有一定距离。
            *   例: "SWOT" vs "TOWS矩阵"; "4P" vs "7P营销组合"。
            *   **业务含义**: 存在理论渊源，但可能进行了改编。
    *   **处理动作**: 仅当 AI 判定为上述三类之一时，才建立映射关系。若无关，则视为“无匹配”（即原创/新颖理论）。
*   **4.2 查重与聚类逻辑 (Future Clustering)**:
    *   **引用**: `BATCH_MATCHING_OPTIMIZATION.md`。
    *   **隐含理论聚类**: 对于 `Implied Theory`，即使名称不同，如果其 `context` 的语义向量 (Embedding) 高度相似，则视为同一理论应用。这需要后续引入向量数据库 (如 FAISS 或 Chroma) 进行语义聚类。

## 5. 开发环境原子配置 (Environment Atoms)
*   **环境变量**:
    *   `API_KEY`: 目前硬编码在 `build_knowledge_base_v2.py` 中，建议迁移至 `.env`。
    *   `PROXY_URL`: 必须设置为 `http://127.0.0.1:7897` (或其他本地代理端口) 以连接 Google API。
    *   `ROOT_DIR`: 默认指向 `C:\Users\LXG\CaseTheoryCheck`。
*   **编码规范 (Encoding Standards)**:
    *   **Python I/O**: 打开文件必须显式指定 `encoding='utf-8'`。
    *   **Console**: 注意 Windows CMD 的 `gbk` 默认编码可能导致 Emoji 或生僻字打印报错，建议设置 `PYTHONIOENCODING=utf-8`。
*   **核心依赖 (Dependencies)**:
    *   `google-generativeai`: Gemini API 客户端。
    *   `pdfplumber`: PDF 文本提取。
    *   `flask` / `fastapi`: 后端服务框架。
    *   `sqlite3`: 标准库，无需安装。

## 6. 调试与运维手册 (Debug & Ops Manual)
*   **日志解读**:
    *   文件: `failed_files_v2.txt`。
    *   格式: `YYYY-MM-DD HH:MM:SS | Filename | Error Reason`。
    *   常见错误: `Deadline Exceeded` (超时 -> 需优化网络或增加 timeout), `Resource Exhausted` (配额耗尽 -> 轮换 Key)。
*   **单点测试 (Isolation Testing)**:
    *   **工具**: 建议复制 `build_knowledge_base_v2.py` 为 `test_single_v2.py`。
    *   **方法**: 修改 `main()` 函数，使其只处理一个指定的 PDF 文件路径，打印 JSON 输出而不写入数据库，以便快速验证 Prompt 效果。
*   **数据库重置 (Reset Procedure)**:
    *   **场景**: Prompt 逻辑发生重大变更 (如新增字段)。
    *   **步骤**: 
        1. 停止所有 Python 进程。
        2. 删除 `knowledge_base_v2.db` 文件。
        3. 重新运行 `python build_knowledge_base_v2.py` (脚本会自动重建表结构)。
