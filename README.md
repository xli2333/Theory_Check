# FDC 商业案例理论查重系统

这是一个智能化的商业案例分析工具，旨在帮助用户识别文稿中与历史案例库的理论框架重合度。系统能够区分显性提及和隐性逻辑，并提供详细的分析报告和修改建议。

## 主要功能

1.  **PDF 文稿解析**：上传 PDF 格式的商业案例文稿，系统将自动提取文本内容。
2.  **AI 理论提取**：利用大型语言模型（Gemini 系列）智能提取文稿中涉及的显性理论框架（直接提及）和隐性逻辑（未直接提及但分析模式一致）。
3.  **三级语义匹配**：将提取出的理论与 FDC 历史案例知识库进行高度重合、次重合、重合三级分类匹配。
4.  **智能总结报告**：生成一份全面的分析报告，包含：
    *   **风险等级评估**：综合评估文稿的原创度风险（高风险、逻辑风险、中度风险、良好）。
    *   **显性与隐性统计**：详细展示显性理论和隐性逻辑的重合数量。
    *   **历史引用趋势**：通过图表展示不同年份历史案例的引用频率。
    *   **重复度最高的历史案例**：列出与当前文稿重复度最高的历史案例。
    *   **AI 判定理由**：针对每个重合理论，AI 会提供简练的判定理由。
    *   **修改建议**：根据风险等级提供针对性的优化建议。
5.  **HTML 报告导出**：支持将详细的分析报告导出为美观且易于打印的 HTML 文件。

## 技术栈

*   **后端**：Python (FastAPI, PDFPlumber, Google Generative AI, SQLite3)
*   **前端**：React (Vite, TypeScript, TailwindCSS, Framer Motion)
*   **数据库**：SQLite (用于存储 FDC 历史知识点)

## 部署与运行

### 后端

1.  **安装依赖**：
    ```bash
    pip install -r requirements.txt
    ```
    （如果 `requirements.txt` 不存在，请根据 `server.py` 中的 `import` 语句手动安装 `fastapi`, `uvicorn`, `google-generativeai`, `pdfplumber`, `pydantic`, `python-dotenv` 等库）
2.  **配置 API Key 和代理**：
    在 `server.py` 中更新您的 Gemini API Key 和可选的代理设置。
    ```python
    API_KEY = "YOUR_GEMINI_API_KEY"
    PROXY_URL = "http://127.0.0.1:7897" # 如果需要
    ```
3.  **运行后端服务**：
    ```bash
    python server.py
    ```
    服务将运行在 `http://127.0.0.1:8000`。

### 前端

1.  **进入前端目录**：
    ```bash
    cd frontend
    ```
2.  **安装依赖**：
    ```bash
    npm install
    ```
3.  **运行前端服务**：
    ```bash
    npm run dev
    ```
    前端应用通常运行在 `http://localhost:5173` 或其他端口。

## 使用方法

1.  确保后端和前端服务都在运行。
2.  在浏览器中打开前端地址 (例如 `http://localhost:5173`)。
3.  上传您要分析的 PDF 文稿。
4.  等待系统完成分析，查看结果报告。
5.  您可以点击“导出打印版报告”按钮下载 HTML 格式的报告。

---

**FDC 智能案例评审系统**
