import os
import sqlite3
import json
import time
import pandas as pd
import pdfplumber
import google.generativeai as genai

# ================= 配置区域 =================
API_KEY = "AIzaSyDaDMWEEp5Dx3FReUyDYcL92aWcNn8jmLI" 
MODEL_NAME = 'gemini-2.5-flash'
DB_PATH = "knowledge_base.db"
PROXY_URL = "http://127.0.0.1:7897" 

# [重要] 请在这里修改您要查重的新文件路径
TARGET_PDF = r"C:\Users\LXG\CaseTheoryCheck\FDC-21案例统计原文\北谷电子有限公司：掘金工业互联网.pdf"

# 输出报告名称
REPORT_FILE = "查重报告_Result.xlsx"
# ===========================================

# 强制代理
os.environ['HTTP_PROXY'] = PROXY_URL
os.environ['HTTPS_PROXY'] = PROXY_URL
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

def extract_text_from_pdf(pdf_path):
    print(f"Reading PDF: {pdf_path}...")
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t: text += t + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None
    return text

def extract_new_concepts(content):
    """提取新文件的知识点，逻辑与建库时一致"""
    print("AI extracting concepts from new file...")
    prompt = f"""
    Analyze the following business case text.
    Task: Extract ALL [Theories] and [Knowledge Points].
    
    Output JSON format:
    {{
        "meta": {{\"title\": "...", \"date\": "..."}},
        "items": [
            {{\"category\": "理论", \"standard_name\": "Name", \"context\": "Context..."}},
            {{\"category\": "知识点", \"standard_name\": "Name", \"context\": "Context..."}}
        ]
    }}
    
    Rule: Better to have too many than too few.
    
    Text:
    {content}
    """
    try:
        res = model.generate_content(prompt)
        raw = res.text.strip()
        if raw.startswith("```"):
            start = raw.find('{')
            end = raw.rfind('}') + 1
            raw = raw[start:end]
        return json.loads(raw)
    except Exception as e:
        print(f"Extraction failed: {e}")
        return None

def get_db_tags(category):
    """从数据库获取指定分类的所有历史标签"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # 获取去重后的标准名
    c.execute("SELECT DISTINCT standard_name FROM knowledge_points WHERE category=?", (category,))
    tags = [row[0] for row in c.fetchall()]
    conn.close()
    return tags

def ai_semantic_match(new_items, db_tags, category_name):
    """
    核心：让 AI 进行语义比对和分级
    """
    if not new_items or not db_tags:
        return []

    # 提取新文件中的标准名列表
    new_tag_list = list(set([item['standard_name'] for item in new_items]))
    
    print(f"AI Matching for [{category_name}]: Comparing {len(new_tag_list)} new terms against {len(db_tags)} DB terms...")

    prompt = f"""
    You are a plagiarism checking judge.
    
    Input:
    1. NEW_TERMS: {json.dumps(new_tag_list, ensure_ascii=False)}
    2. DB_TAGS: {json.dumps(db_tags, ensure_ascii=False)}
    
    Task:
    For each term in NEW_TERMS, check if there is a semantic match in DB_TAGS.
    
    Ranking Criteria (relevance):
    - "High" (最相关): Exact synonyms, same concept, direct translation (e.g., "Long Tail" vs "Long Tail Theory").
    - "Medium" (次相关): Strong relationship, containment, or closely related concepts (e.g., "Private Traffic" vs "User Retention").
    - "Low" (相关): Loosely related, same broad field, worth checking (e.g., "AI" vs "Big Data").
    
    Output:
    Return a JSON LIST of matches. If no match found for a term, do not include it.
    
    JSON Example:
    [
        {{"new_term": "KOL营销", "db_term": "关键意见领袖", "relevance": "High", "reason": "Synonyms"}},
        {{"new_term": "数字化", "db_term": "企业转型", "relevance": "Medium", "reason": "Related concept"}}
    ]
    """
    
    try:
        # 给足时间思考
        res = model.generate_content(prompt, request_options={'timeout': 600})
        raw = res.text.strip()
        if raw.startswith("```"):
            start = raw.find('[')
            end = raw.rfind(']') + 1
            raw = raw[start:end]
        return json.loads(raw)
    except Exception as e:
        print(f"Matching failed: {e}")
        return []

def generate_report_data(matches, new_items, category):
    """
    根据匹配结果，回查数据库，生成表格行
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    report_rows = []
    
    # 将新文件的 item 转为字典方便查询上下文
    new_item_map = {item['standard_name']: item['context'] for item in new_items}
    
    for match in matches:
        new_term = match['new_term']
        db_term = match['db_term']
        relevance = match['relevance']
        
        # 1. 获取新文件的上下文
        new_context = new_item_map.get(new_term, "")
        
        # 2. 获取旧文件的证据 (可能有多篇)
        # 查重时我们要看：这个 db_term 在库里出现在哪些文章里？
        query = '''
            SELECT f.filename, f.publish_date, k.context 
            FROM knowledge_points k
            JOIN files f ON k.file_id = f.id
            WHERE k.standard_name = ? AND k.category = ?
        '''
        c.execute(query, (db_term, category))
        history_records = c.fetchall()
        
        for record in history_records:
            filename, date, old_context = record
            report_rows.append({
                "相关度": relevance, # High/Medium/Low
                "新文件关键词": new_term,
                "新文件语境": new_context,
                "匹配库中关键词": db_term,
                "历史来源文件": filename,
                "年份": date,
                "历史语境证据": old_context
            })
            
    conn.close()
    return report_rows

def main():
    if not os.path.exists(TARGET_PDF):
        print(f"File not found: {TARGET_PDF}")
        return

    # 1. 读取并提取新文件
    text = extract_text_from_pdf(TARGET_PDF)
    if not text: return
    
    new_data = extract_new_concepts(text)
    if not new_data: return
    
    all_new_items = new_data.get('items', [])
    # 分拆新文件内容
    new_theories = [i for i in all_new_items if i['category'] == '理论']
    new_points = [i for i in all_new_items if i['category'] == '知识点']
    
    # 2. 准备结果容器
    writer = pd.ExcelWriter(REPORT_FILE, engine='openpyxl')
    
    # --- 处理理论部分 ---
    db_theory_tags = get_db_tags('理论')
    print(f"Found {len(db_theory_tags)} unique Theories in DB.")
    theory_matches = ai_semantic_match(new_theories, db_theory_tags, "理论")
    theory_rows = generate_report_data(theory_matches, new_theories, "理论")
    
    if theory_rows:
        df_theory = pd.DataFrame(theory_rows)
        # 按相关度排序 (自定义排序 High > Medium > Low)
        relevance_order = {"High": 1, "Medium": 2, "Low": 3}
        df_theory['sort_key'] = df_theory['相关度'].map(relevance_order)
        df_theory = df_theory.sort_values('sort_key').drop('sort_key', axis=1)
        df_theory.to_excel(writer, sheet_name='理论查重', index=False)
    else:
        print("No Theory matches found.")

    # --- 处理知识点部分 ---
    db_point_tags = get_db_tags('知识点')
    print(f"Found {len(db_point_tags)} unique Knowledge Points in DB.")
    point_matches = ai_semantic_match(new_points, db_point_tags, "知识点")
    point_rows = generate_report_data(point_matches, new_points, "知识点")
    
    if point_rows:
        df_point = pd.DataFrame(point_rows)
        # 排序
        relevance_order = {"High": 1, "Medium": 2, "Low": 3}
        df_point['sort_key'] = df_point['相关度'].map(relevance_order)
        df_point = df_point.sort_values('sort_key').drop('sort_key', axis=1)
        df_point.to_excel(writer, sheet_name='知识点查重', index=False)
    else:
        print("No Knowledge Point matches found.")
        
    writer.close()
    print(f"\nReport generated: {os.path.abspath(REPORT_FILE)}")

if __name__ == "__main__":
    main()
