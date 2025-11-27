import os
import glob
import pandas as pd
import pdfplumber
import google.generativeai as genai
import time
import json
import re
import sys

# ================= 配置区域 =================
# 您的 API Key
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    print("[Error] API_KEY environment variable not set.")
    sys.exit(1)

# 模型名称 (如果您指的是 2.0 Flash 预览版，请改为 'gemini-2.0-flash-exp')
MODEL_NAME = 'gemini-2.5-flash'

# 目标根目录 (当前目录)
ROOT_DIR = r"C:\Users\LXG\CaseTheoryCheck"

# 输出文件路径
OUTPUT_CSV = "FDC_Case_Analysis_Result.csv"
# ===========================================

# 配置 Gemini
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

def extract_text_from_pdf(pdf_path):
    """提取 PDF 文本"""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # 限制页数以防极个别超大文件卡死，但通常 Flash 1.5 上下文很大，全读没问题
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"[Error] Reading PDF failed: {pdf_path} - {e}")
        return None
    return text

def analyze_content(filename, content):
    """调用 Gemini 进行分析，强制 JSON 输出"""
    
    # 提示词工程：明确要求 JSON 格式，且包含 Context 佐证
    prompt = f"""
    You are an expert in business management case analysis. Analyze the following text content from the file: "{filename}".
    
    Your Task:
    1. Extract the [Title] and [Date] (if mentioned in the text).
    2. Extract ALL [Knowledge Points] (Practical concepts, industry terms, management methods).
    3. Extract ALL [Theories] (Academic models, frameworks, laws, e.g., SWOT, Long Tail, Porter's Five Forces).
    4. Provide the [Context] for each point: Explain HOW and WHY this concept/theory is involved in the text. 
    
    Constraints:
    - "Better to have too many than too few" (宁缺毋滥): If a concept is mentioned or applied, include it.
    - OUTPUT FORMAT: Strictly a JSON list of objects. No Markdown formatting.
    - Language: Simplified Chinese (简体中文).
    
    JSON Structure Example:
    [
        {{
            "title": "Case Title Here",
            "date": "2021-05-24",
            "type": "理论", 
            "name": "长尾理论",
            "context": "文中分析了中小企业如何利用长尾市场进行差异化竞争..."
        }},
        {{
            "title": "Case Title Here",
            "date": "2021-05-24",
            "type": "知识点", 
            "name": "B2B电商",
            "context": "案例详细描述了公司从B2B平台转型的过程..."
        }}
    ]

    Text Content:
    {content}
    """

    try:
        # 尝试生成
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        
        # 清洗 Markdown 标记 (```json ... ```)
        if raw_text.startswith("```"):
            # 找到第一个 [ 和最后一个 ]
            start_idx = raw_text.find('[')
            end_idx = raw_text.rfind(']') + 1
            if start_idx != -1 and end_idx != -1:
                raw_text = raw_text[start_idx:end_idx]
            else:
                # 兜底：如果没找到 []，可能格式不对
                return None

        return json.loads(raw_text)
    except Exception as e:
        print(f"[AI Error] Analysis failed for {filename}: {e}")
        return None

def main():
    print(f"--- Starting Analysis Pipeline using {MODEL_NAME} ---")
    print(f"Scanning directory: {ROOT_DIR}")

    # 1. 扫描所有 PDF
    all_pdfs = []
    for root, dirs, files in os.walk(ROOT_DIR):
        for file in files:
            if file.lower().endswith(".pdf"):
                all_pdfs.append(os.path.join(root, file))
    
    total_files = len(all_pdfs)
    print(f"Found {total_files} PDF files.")

    # 2. 读取已处理的文件（断点续传）
    processed_files = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            # 尝试读取，如果有编码错误则尝试不同编码
            try:
                df_existing = pd.read_csv(OUTPUT_CSV, encoding='utf-8-sig')
            except:
                df_existing = pd.read_csv(OUTPUT_CSV, encoding='gbk')
                
            if 'filename' in df_existing.columns:
                processed_files = set(df_existing['filename'].unique())
            print(f"Resuming... {len(processed_files)} files already processed. {total_files - len(processed_files)} remaining.")
        except Exception as e:
            print(f"[Warning] Could not read existing CSV ({e}), starting fresh.")

    # 3. 循环处理
    for index, pdf_path in enumerate(all_pdfs):
        filename = os.path.basename(pdf_path)
        
        # 进度条效果
        sys.stdout.write(f"\rProgress: [{index+1}/{total_files}] Current: {filename[:30]}...")
        sys.stdout.flush()

        if filename in processed_files:
            continue

        # 提取文本
        content = extract_text_from_pdf(pdf_path)
        if not content or len(content) < 50:
            print(f"\n[Skip] {filename} (Content too short or empty)")
            continue

        # AI 分析 (简单的重试机制)
        result_json = None
        for attempt in range(3):
            result_json = analyze_content(filename, content)
            if result_json:
                break
            time.sleep(2) # 失败等待2秒重试

        if not result_json:
            print(f"\n[Failed] Could not analyze {filename} after 3 attempts.")
            continue

        # 转换为 DataFrame 并追加写入
        new_rows = []
        for item in result_json:
            new_rows.append({
                "filename": filename,
                "title": item.get("title", ""),
                "date": item.get("date", ""),
                "type": item.get("type", "未分类"),
                "name": item.get("name", ""),
                "context": item.get("context", "")
            })
        
        if new_rows:
            df_new = pd.DataFrame(new_rows)
            # 检查是否需要写表头
            header = not os.path.exists(OUTPUT_CSV)
            try:
                df_new.to_csv(OUTPUT_CSV, mode='a', header=header, index=False, encoding='utf-8-sig')
            except Exception as write_err:
                 print(f"\n[Error] Could not write to CSV: {write_err}")
        
        # 遵守 API 速率限制 (Flash 很高，但稍微停顿更安全)
        time.sleep(1)

    print(f"\n\n--- Done! Results saved to {OUTPUT_CSV} ---")

if __name__ == "__main__":
    main()
