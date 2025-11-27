import os
import sqlite3
import json
import time
import pandas as pd
import pdfplumber
import google.generativeai as genai
from datetime import datetime

# ================= 配置区域 =================
API_KEY = "AIzaSyDaDMWEEp5Dx3FReUyDYcL92aWcNn8jmLI" 
MODEL_NAME = 'gemini-2.5-flash'
DB_PATH = "knowledge_base.db"
PROXY_URL = "http://127.0.0.1:7897" 

# 目标文件 (请按需修改)
TARGET_PDF = r"C:\Users\LXG\CaseTheoryCheck\FDC-21案例统计原文\北谷电子有限公司：掘金工业互联网.pdf"

# 输出文件
REPORT_JSON = "report_data.json"
REPORT_EXCEL = "查重结果明细.xlsx"
REPORT_HTML = "Final_Report.html"
# ===========================================

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
    print("AI extracting concepts from new file...")
    prompt = f"""
    Analyze the following business case text.
    Task: Extract ALL [Theories] and [Knowledge Points].
    
    Output JSON format:
    {{
        "meta": {{\"title\": "Detect Title", "date": "Detect Date"}},
        "items": [
            {{\"category\": "理论", "standard_name\": "Name", "context\": "Context..."}},
            {{\"category\": "知识点", "standard_name\": "Name", "context\": "Context..."}}
        ]
    }}
    
    Rule: Better to have too many than too few.
    AMBIGUITY: If unsure, output in BOTH categories.
    
    Text:
    {content}
    """
    try:
        res = model.generate_content(prompt, request_options={'timeout': 600})
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
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT DISTINCT standard_name FROM knowledge_points WHERE category=?", (category,))
    tags = [row[0] for row in c.fetchall()]
    conn.close()
    return tags

def ai_semantic_match(new_items, db_tags, category_name):
    if not new_items or not db_tags:
        return []
    
    new_tag_list = list(set([item['standard_name'] for item in new_items]))
    print(f"AI Matching for [{category_name}]: {len(new_tag_list)} vs {len(db_tags)} terms...")

    prompt = f"""
    You are a plagiarism checking judge.
    
    Input:
    1. NEW_TERMS: {json.dumps(new_tag_list, ensure_ascii=False)}
    2. DB_TAGS: {json.dumps(db_tags, ensure_ascii=False)}
    
    Task: Find semantic matches.
    Ranking:
    - "High": Synonyms, exact concept.
    - "Medium": Strong relation, containment.
    - "Low": Loose relation, same field. 
    
    Output JSON LIST:
    [
        {{"new_term": "A", "db_term": "B", "relevance": "High", "reason": "..."}}
    ]
    """
    
    try:
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

def build_report_data(matches, new_items, category):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    results = []
    new_item_map = {item['standard_name']: item['context'] for item in new_items}
    
    for match in matches:
        new_term = match['new_term']
        db_term = match['db_term']
        
        # 获取历史证据
        c.execute('''
            SELECT f.filename, f.publish_date, k.context 
            FROM knowledge_points k
            JOIN files f ON k.file_id = f.id
            WHERE k.standard_name = ? AND k.category = ?
        ''', (db_term, category))
        
        evidence = []
        for row in c.fetchall():
            evidence.append({
                "filename": row[0],
                "year": row[1],
                "context": row[2]
            })
            
        results.append({
            "risk_level": match['relevance'],
            "new_term": new_term,
            "new_context": new_item_map.get(new_term, ""),
            "db_term": db_term,
            "reason": match.get('reason', ''),
            "evidence": evidence
        })
    
    conn.close()
    return results

def generate_html(report_data):
    """
    生成高级商业风格的 HTML 报告
    """
    print("Generating HTML Report...")
    
    # 统计数据
    total_theories = len(report_data['results']['theories'])
    total_points = len(report_data['results']['knowledge_points'])
    high_risks = sum(1 for i in report_data['results']['theories'] + report_data['results']['knowledge_points'] if i['risk_level'] == 'High')
    
    # 获取当前时间
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>FDC 商业案例查重报告</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            @media print {{
                body {{ -webkit-print-color-adjust: exact; }}
                .page-break {{ page-break-before: always; }}
                .no-print {{ display: none; }}
            }}
            body {{ font-family: 'Inter', sans-serif; background-color: #f8fafc; color: #1e293b; }}
            .serif {{ font-family: 'Noto Serif SC', serif; }}
            .risk-High {{ color: #e11d48; font-weight: bold; }}
            .risk-Medium {{ color: #d97706; font-weight: bold; }}
            .risk-Low {{ color: #64748b; font-weight: bold; }}
            
            .badge-High {{ background-color: #fee2e2; color: #991b1b; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; border: 1px solid #fecaca; }}
            .badge-Medium {{ background-color: #fef3c7; color: #92400e; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; border: 1px solid #fde68a; }}
            .badge-Low {{ background-color: #f1f5f9; color: #475569; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; border: 1px solid #e2e8f0; }}
        </style>
    </head>
    <body class="p-0 m-0">

        <!-- 封面 -->
        <div class="min-h-screen flex flex-col justify-center items-center bg-white border-b-4 border-slate-800 p-12 text-center">
            <div class="mb-8">
                <svg class="w-20 h-20 text-slate-800 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
            </div>
            <h1 class="text-4xl font-bold tracking-tight text-slate-900 mb-4 serif">商业案例查重分析报告</h1>
            <p class="text-xl text-slate-500 mb-12">FDC Business Case Duplication Analysis</p>
            
            <div class="w-full max-w-2xl bg-slate-50 rounded-xl border border-slate-200 p-8 text-left shadow-sm">
                <div class="grid grid-cols-2 gap-6">
                    <div>
                        <p class="text-sm text-slate-400 uppercase tracking-wider font-semibold">目标文件</p>
                        <p class="text-lg font-medium text-slate-900 truncate" title="{report_data['meta']['target_filename']}">{report_data['meta']['target_filename']}</p>
                    </div>
                    <div>
                        <p class="text-sm text-slate-400 uppercase tracking-wider font-semibold">检测时间</p>
                        <p class="text-lg font-medium text-slate-900">{gen_time}</p>
                    </div>
                    <div>
                        <p class="text-sm text-slate-400 uppercase tracking-wider font-semibold">检测范围</p>
                        <p class="text-lg font-medium text-slate-900">FDC-21 至 FDC-25 历史库</p>
                    </div>
                     <div>
                        <p class="text-sm text-slate-400 uppercase tracking-wider font-semibold">高风险项</p>
                        <p class="text-2xl font-bold text-rose-600">{high_risks}</p>
                    </div>
                </div>
            </div>
            
            <div class="mt-20 text-slate-400 text-sm">
                <p>Generated by FDC AI Analysis System</p>
            </div>
        </div>

        <!-- 概览页 -->
        <div class="page-break max-w-5xl mx-auto p-12 bg-white min-h-screen">
            <h2 class="text-2xl font-bold border-b border-slate-200 pb-4 mb-8 text-slate-800">1. 检测概览 (Executive Summary)</h2>
            
            <div class="grid grid-cols-3 gap-6 mb-12">
                <div class="bg-slate-50 p-6 rounded-lg border border-slate-100">
                    <p class="text-sm text-slate-500 mb-1">理论框架重复点</p>
                    <p class="text-3xl font-bold text-slate-800">{total_theories}</p>
                </div>
                <div class="bg-slate-50 p-6 rounded-lg border border-slate-100">
                    <p class="text-sm text-slate-500 mb-1">实务知识点重复点</p>
                    <p class="text-3xl font-bold text-slate-800">{total_points}</p>
                </div>
                <div class="bg-rose-50 p-6 rounded-lg border border-rose-100">
                    <p class="text-sm text-rose-600 mb-1">综合高风险预警</p>
                    <p class="text-3xl font-bold text-rose-700">{high_risks}</p>
                </div>
            </div>

            <div class="prose max-w-none text-slate-600">
                <p>本报告基于语义分析技术，对目标文件进行了全量的知识点提取与比对。系统共检测到 <strong>{total_theories + total_points}</strong> 处潜在的相似内容。</p>
                <p class="mt-4">
                    <span class="font-semibold text-slate-900">图例说明：</span><br>
                    <span class="inline-block w-3 h-3 bg-rose-500 rounded-full mr-2"></span><span class="font-bold text-slate-800">High (高度相关)</span>: 完全同义、直接引用或核心概念一致，建议重点排查。<br>
                    <span class="inline-block w-3 h-3 bg-amber-500 rounded-full mr-2"></span><span class="font-bold text-slate-800">Medium (次相关)</span>: 包含关系或强依赖关系，属于常规引用或路径依赖。<br>
                    <span class="inline-block w-3 h-3 bg-slate-400 rounded-full mr-2"></span><span class="font-bold text-slate-800">Low (相关)</span>: 同属于一个大领域，通常为行业背景介绍。
                </p>
            </div>
        </div>

        <!-- 理论详情页 -->
        <div class="page-break max-w-5xl mx-auto p-12 bg-white">
            <h2 class="text-2xl font-bold border-b border-slate-200 pb-4 mb-8 text-slate-800">2. 理论框架查重详情 (Theoretical Framework)</h2>
            {generate_section_html(report_data['results']['theories'])}
        </div>

        <!-- 知识点详情页 -->
        <div class="page-break max-w-5xl mx-auto p-12 bg-white">
            <h2 class="text-2xl font-bold border-b border-slate-200 pb-4 mb-8 text-slate-800">3. 实务知识点查重详情 (Practical Concepts)</h2>
            {generate_section_html(report_data['results']['knowledge_points'])}
        </div>

    </body>
    </html>
    ""
    
    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML saved to {REPORT_HTML}")

def generate_section_html(items):
    if not items:
        return '<p class="text-slate-500 italic">未检测到相关重复项。</p>'
    
    # 排序：High -> Medium -> Low
    order = {'High': 0, 'Medium': 1, 'Low': 2}
    items.sort(key=lambda x: order.get(x['risk_level'], 3))
    
    html = '<div class="space-y-8">'
    for idx, item in enumerate(items):
        badge_class = f"badge-{item['risk_level']}"
        
        evidence_html = ""
        # 只显示前3个证据，避免太长
        for ev in item['evidence'][:3]:
            evidence_html += f"""
            <div class="mt-3 pl-4 border-l-2 border-slate-200">
                <p class="text-xs font-bold text-slate-700">{ev['filename']} <span class="text-slate-400 font-normal">({ev['year']})</span></p>
                <p class="text-sm text-slate-500 italic mt-1 serif">“{ev['context'][:150]}...”</p>
            </div>
            """

        html += f"""
        <div class="border border-slate-200 rounded-lg p-6 bg-white shadow-sm break-inside-avoid">
            <div class="flex justify-between items-start mb-4">
                <div>
                    <span class="text-slate-400 text-sm font-mono mr-2">#{idx+1}</span>
                    <span class="text-lg font-bold text-slate-900">{item['new_term']}</span>
                    <span class="mx-2 text-slate-300">→</span>
                    <span class="text-sm font-semibold text-slate-600">{item['db_term']}</span>
                </div>
                <span class="{badge_class}">{item['risk_level']} Risk</span>
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <!-- 左侧：新文 -->
                <div class="bg-slate-50 p-4 rounded text-sm">
                    <p class="text-xs text-slate-400 uppercase tracking-wider font-semibold mb-2">当前文稿语境</p>
                    <p class="text-slate-700 leading-relaxed serif">{item['new_context']}</p>
                </div>
                
                <!-- 右侧：证据 -->
                <div>
                    <p class="text-xs text-slate-400 uppercase tracking-wider font-semibold mb-2">历史案例库证据</p>
                    {evidence_html}
                </div>
            </div>
            
            <div class="mt-3 text-xs text-slate-400 flex items-center">
                <svg class="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                判定理由: {item['reason']}
            </div>
        </div>
        """
    html += '</div>'
    return html

def main():
    print("--- FDC Check System Final ---")
    if not os.path.exists(TARGET_PDF):
        print("Target file not found.")
        return
        
    # 1. 提取
    text = extract_text_from_pdf(TARGET_PDF)
    new_data = extract_new_concepts(text)
    if not new_data: return
    
    all_new_items = new_data.get('items', [])
    new_theories = [i for i in all_new_items if i['category'] == '理论']
    new_points = [i for i in all_new_items if i['category'] == '知识点']
    
    # 2. 匹配
    db_theories = get_db_tags('理论')
    db_points = get_db_tags('知识点')
    
    matches_theory = ai_semantic_match(new_theories, db_theories, '理论')
    matches_point = ai_semantic_match(new_points, db_points, '知识点')
    
    # 3. 构建结果数据
    theory_results = build_report_data(matches_theory, new_theories, '理论')
    point_results = build_report_data(matches_point, new_points, '知识点')
    
    full_report = {
        "meta": {
            "target_filename": os.path.basename(TARGET_PDF),
            "generated_at": datetime.now().isoformat()
        },
        "results": {
            "theories": theory_results,
            "knowledge_points": point_results
        }
    }
    
    # 4. 输出 JSON
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    
    # 5. 输出 Excel (可选，作为附件)
    # 将 JSON 展平为 DataFrame
    rows = []
    for cat, items in [("理论", theory_results), ("知识点", point_results)]:
        for item in items:
            rows.append({
                "赛道": cat,
                "风险等级": item['risk_level'],
                "新词": item['new_term'],
                "旧词": item['db_term'],
                "理由": item['reason']
            })
    pd.DataFrame(rows).to_excel(REPORT_EXCEL, index=False)
    
    # 6. 输出 HTML
    generate_html(full_report)
    
    print("\n=== 全部完成 ===")
    print(f"1. 前端数据: {REPORT_JSON}")
    print(f"2. 审核表格: {REPORT_EXCEL}")
    print(f"3. 打印报告: {REPORT_HTML} (请在浏览器打开并打印为PDF)")

if __name__ == "__main__":
    main()
