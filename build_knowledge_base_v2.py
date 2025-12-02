import os
import sqlite3
import json
import time
import sys
import pdfplumber
import google.generativeai as genai
from datetime import datetime

# ================= 配置区域 =================
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = 'gemini-2.5-flash'
ROOT_DIR = os.environ.get("ROOT_DIR", r"C:\Users\LXG\CaseTheoryCheck") # Default to local if not set
DB_PATH = "knowledge_base_v2.db"
FAILED_LOG = "failed_files_v2.txt" 
PROXY_URL = os.environ.get("PROXY_URL", "http://127.0.0.1:7897") # Default to local proxy
# ===========================================

# 1. 强制代理 (仅当 PROXY_URL 存在且不在 Render 上时)
if not os.environ.get('RENDER') and PROXY_URL:
    os.environ['HTTP_PROXY'] = PROXY_URL
    os.environ['HTTPS_PROXY'] = PROXY_URL

# 2. 配置 Gemini
if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    print("Warning: API_KEY not set in environment variables.")

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
    
    # Modified schema: 'category' will store 'Direct Theory' or 'Implied Theory'
    # 'standard_name' stores the theory name
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS theories (
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
    You are an expert academic researcher. Analyze the text from the case file "{filename}".
    
    Your SOLE GOAL is to extract **Theories** (Academic models, frameworks, laws, theoretical concepts).
    Do NOT extract general knowledge points, industry terms, or practical strategies unless they are actual theories.

    You must categorize each extracted theory into one of two types:
    
    1. **Direct Theory (直接理论)**: 
       - The theory is EXPLICITLY mentioned by name in the text.
       - Example: The text says "We used SWOT analysis...", you extract "SWOT Analysis".
       
    2. **Implied Theory (隐含理论)**: 
       - The theory is NOT explicitly named, but the text describes a logic, process, or framework that clearly corresponds to a specific academic theory.
       - Example: The text describes "analyzing strengths, weaknesses, opportunities, and threats" without naming SWOT. You extract "SWOT Analysis" as an Implied Theory.
       - Example: The text discusses "users weighing perceived usefulness against ease of use", you extract "Technology Acceptance Model (TAM)" as an Implied Theory.

    For each extracted theory, you must provide:
    - **category**: Either "Direct Theory" or "Implied Theory".
    - **standard_name**: The standard academic name of the theory (e.g., "Agenda Setting Theory", "Uses and Gratifications Theory").
    - **original_text**: The exact text segment that mentions or implies the theory.
    - **context**: A brief excerpt (1-2 sentences) of the surrounding text to prove why you extracted this.

    OUTPUT FORMAT:
    Return ONLY a JSON object. No Markdown.
    {{
        "meta": {{
            "title": "Article Title",
            "date": "YYYY-MM-DD"
        }},
        "items": [
            {{
                "category": "Direct Theory",
                "standard_name": "Agenda Setting Theory",
                "original_text": "议程设置理论",
                "context": "文中提到...'根据议程设置理论，媒体通过...'..."
            }},
            {{
                "category": "Implied Theory",
                "standard_name": "Framing Theory",
                "original_text": "强调特定方面而忽略其他",
                "context": "文中描述...'报道着重突出了事件的负面影响，构建了特定的解读框架'..."
            }}
        ]
    }}

    Text Content (Truncated for processing if necessary):
    {content[:30000]} 
    """
    # Note: Added content truncation to avoid token limits if files are huge, 
    # though 2.5-flash has a large context window, it's safer for 'context' extraction quality.
    # User asked for "Retain corresponding context", full text is better if possible.
    # 2.5 flash has 1M context, so truncation is likely unnecessary for typical PDFs.
    # I will remove the specific slice and pass 'content' but keep an eye on errors.
    
    final_prompt = prompt.replace("{content[:30000]}", content)

    try:
        response = model.generate_content(final_prompt, request_options={'timeout': 600})
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
                    item.get("category", "Uncategorized"),
                    item.get("standard_name", ""),
                    item.get("original_text", ""),
                    item.get("context", "")
                ))
            
            cursor.executemany('''
                INSERT INTO theories (file_id, category, standard_name, original_text, context)
                VALUES (?, ?, ?, ?, ?)
            ''', rows)
            
        conn.commit()
        return len(items)
        
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
    print(f"--- Knowledge Base Builder V2 (Theories Only) ---")
    init_db()
    
    all_pdfs = []
    # Specifically looking for FDC-21 to FDC-25 folders as per user intent context
    # But sticking to ROOT_DIR walk to be generic yet covering.
    for root, dirs, files in os.walk(ROOT_DIR):
        for file in files:
            if file.lower().endswith(".pdf"):
                # Optional: Filter for FDC paths if needed, but processing all is safer default
                # unless explicitly restricted. User said "process 21-25 original text".
                # I'll check if the path contains 'FDC-2' to be sure we are targeting the right dataset
                # to avoid re-processing unrelated PDFs if any.
                if "FDC-2" in root or "FDC-" in root: 
                    all_pdfs.append(os.path.join(root, file))
    
    print(f"Found {len(all_pdfs)} PDF files in FDC-* directories.")
    
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
        
        end_time = time.time()
        duration = end_time - start_time
        
        if success:
            print(f" -> [Success] Saved {saved_count} theories ({duration:.1f}s)")
        else:
            print(f"\n[FAILED] {filename} after 3 attempts.")
            log_failure(filename, "Analysis Failed after 3 retries")
            
        time.sleep(1) 

    print("\n--- Build Complete ---")
    if os.path.exists(FAILED_LOG):
        print(f"Check {FAILED_LOG} for any failed files.")

if __name__ == "__main__":
    main()
