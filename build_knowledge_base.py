import os
import sqlite3
import json
import time
import sys
import pdfplumber
import google.generativeai as genai
from datetime import datetime

# ================= 配置区域 =================
API_KEY = "AIzaSyDaDMWEEp5Dx3FReUyDYcL92aWcNn8jmLI" 
MODEL_NAME = 'gemini-2.5-flash'
ROOT_DIR = r"C:\Users\LXG\CaseTheoryCheck"
DB_PATH = "knowledge_base.db"
FAILED_LOG = "failed_files.txt" 
PROXY_URL = "http://127.0.0.1:7897" 
# ===========================================

# 1. 强制代理
os.environ['HTTP_PROXY'] = PROXY_URL
os.environ['HTTPS_PROXY'] = PROXY_URL

# 2. 配置 Gemini
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            filepath TEXT,
            title TEXT,
            publish_date TEXT,
            processed_at TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge_points (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER,
            category TEXT,
            standard_name TEXT,
            original_text TEXT,
            context TEXT,
            FOREIGN KEY(file_id) REFERENCES files(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"\n[Error] Reading PDF {pdf_path}: {e}")
        return None
    return text

def analyze_and_extract(filename, content):
    prompt = f"""
    You are a strictly compliant data extraction assistant. Analyze the text from file "{filename}".
    
    Goals:
    1. Extract the [Title] and [Date].
    2. Extract ALL [Theories] (Academic models, frameworks, laws).
    3. Extract ALL [Knowledge Points] (Industry terms, practical concepts, strategies).
    
    CRITICAL RULE - "Better to have too many than too few" (宁缺毋滥): 
    - Even if a concept is only briefly mentioned, EXTRACT IT.
    
    *** IMPORTANT: AMBIGUITY HANDLING ***
    - If a concept is ambiguous or could be classified as BOTH a Theory and a Knowledge Point, **EXTRACT IT TWICE**.
    - Output one entry with category "Theory" and another entry with category "Knowledge Point".

    CRITICAL RULE - Standardization:
    - For each item, provide a "standard_name" (The common, canonical name).
    - Also provide the "original_text" (The exact word/phrase used in the text).

    OUTPUT FORMAT:
    Return ONLY a JSON object. No Markdown.
    {{
        "meta": {{
            "title": "Article Title",
            "date": "YYYY-MM-DD"
        }},
        "items": [
            {{
                "category": "理论",
                "standard_name": "SWOT Analysis",
                "original_text": "SWOT模型",
                "context": "Context..."
            }},
            {{
                "category": "知识点",
                "standard_name": "SWOT Analysis",
                "original_text": "SWOT模型",
                "context": "Context..."
            }}
        ]
    }}

    Text Content:
    {content}
    """
    
    try:
        response = model.generate_content(prompt, request_options={'timeout': 600})
        raw = response.text.strip()
        
        if raw.startswith("```"):
            start = raw.find('{')
            end = raw.rfind('}') + 1
            raw = raw[start:end]
            
        return json.loads(raw)
    except Exception as e:
        raise e 

def save_to_db(filename, filepath, data):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        meta = data.get("meta", {})
        # [修改] 使用 strftime 存入字符串，消除 DeprecationWarning
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute('''
            INSERT INTO files (filename, filepath, title, publish_date, processed_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (filename, filepath, meta.get("title", ""), meta.get("date", ""), now_str))
        
        file_id = cursor.lastrowid
        
        items = data.get("items", [])
        if items:
            rows = []
            for item in items:
                rows.append((
                    file_id,
                    item.get("category", "未分类"),
                    item.get("standard_name", ""),
                    item.get("original_text", ""),
                    item.get("context", "")
                ))
            
            cursor.executemany('''
                INSERT INTO knowledge_points (file_id, category, standard_name, original_text, context)
                VALUES (?, ?, ?, ?, ?)
            ''', rows)
            
        conn.commit()
        return len(items) # 返回保存的数量
        
    except sqlite3.IntegrityError:
        print(f" -> [Skip] Already in DB")
        return 0
    except Exception as e:
        print(f" -> [DB Error] {e}")
        return 0
    finally:
        conn.close()

def log_failure(filename, reason):
    with open(FAILED_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} | {filename} | {reason}\n")

def main():
    print(f"--- Knowledge Base Builder (Timeout: 600s) ---")
    init_db()
    
    all_pdfs = []
    for root, dirs, files in os.walk(ROOT_DIR):
        for file in files:
            if file.lower().endswith(".pdf"):
                all_pdfs.append(os.path.join(root, file))
    
    print(f"Found {len(all_pdfs)} PDF files.")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT filename FROM files")
    processed_files = {row[0] for row in cursor.fetchall()}
    conn.close()
    
    print(f"Skipping {len(processed_files)} already processed files.")
    
    for i, pdf_path in enumerate(all_pdfs):
        filename = os.path.basename(pdf_path)
        
        if filename in processed_files:
            continue
            
        sys.stdout.write(f"Processing [{i+1}/{len(all_pdfs)}]: {filename[:25]}...")
        sys.stdout.flush()
        
        content = extract_text_from_pdf(pdf_path)
        if not content:
            print(" -> Empty/Unreadable")
            log_failure(filename, "Empty or Unreadable PDF")
            continue
            
        # [新增] 计时器开始
        start_time = time.time()
        
        success = False
        saved_count = 0
        
        for attempt in range(3):
            try:
                data = analyze_and_extract(filename, content)
                if data:
                    saved_count = save_to_db(filename, pdf_path, data)
                    success = True
                    break
            except Exception as e:
                error_msg = str(e)
                if "Deadline Exceeded" in error_msg:
                    print(f" [Timeout]", end="")
                else:
                    print(f" [Err: {error_msg[:10]}...]", end="")
                time.sleep(5)
        
        # [新增] 计时器结束与显示
        end_time = time.time()
        duration = end_time - start_time
        
        if success:
            # 打印格式：-> [Success] Saved 177 items (12.5s)
            print(f" -> [Success] Saved {saved_count} items ({duration:.1f}s)")
        else:
            print(f"\n[FAILED] {filename} after 3 attempts.")
            log_failure(filename, "Analysis Failed after 3 retries")
            
        time.sleep(1) 

    print("\n--- Build Complete ---")
    if os.path.exists(FAILED_LOG):
        print(f"Check {FAILED_LOG} for any failed files.")

if __name__ == "__main__":
    main()
