import pdfplumber
import google.generativeai as genai
import json
import pandas as pd
import os

# 配置
API_KEY = "AIzaSyDaDMWEEp5Dx3FReUyDYcL92aWcNn8jmLI"
MODEL_NAME = 'gemini-2.5-flash'
TARGET_FILE = r"C:\Users\LXG\CaseTheoryCheck\FDC-21案例统计原文\IT 赋能科研实验室智慧运维：十衍淘的细分市场转型之路.pdf"
OUTPUT_CSV = "test_result.csv"

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

def extract_text(pdf_path):
    print(f"Reading file: {pdf_path}...")
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None
    return text

def analyze(content):
    print(f"Analyzing with {MODEL_NAME}...")
    prompt = f"""
    You are an expert in business management case analysis. Analyze the following text content.
    
    Task:
    1. Extract [Title] and [Date].
    2. Extract ALL [Knowledge Points] (Practical concepts, industry terms).
    3. Extract ALL [Theories] (Academic models, frameworks).
    4. Provide [Context] for each: HOW and WHY it is involved.

    Constraints:
    - "Better to have too many than too few" (宁缺毋滥).
    - OUTPUT: JSON list only.
    - Language: Simplified Chinese.

    JSON Example:
    [
        {{
            "title": "Title",
            "date": "202x-xx-xx",
            "type": "理论", 
            "name": "Concept Name",
            "context": "Explanation..."
        }}
    ]

    Content:
    {content}
    """
    
    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
        # Simple cleanup
        if raw.startswith("```"):
            start = raw.find('[')
            end = raw.rfind(']') + 1
            raw = raw[start:end]
        return json.loads(raw)
    except Exception as e:
        print(f"AI Error: {e}")
        return None

def main():
    if not os.path.exists(TARGET_FILE):
        print(f"File not found: {TARGET_FILE}")
        return

    text = extract_text(TARGET_FILE)
    if not text:
        return

    data = analyze(text)
    if data:
        df = pd.DataFrame(data)
        print("\n" + "="*30 + " Analysis Result " + "="*30)
        print(df.to_string()) # Print to console
        print("="*80)
        
        # Save to CSV for inspection
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        print(f"\nSaved to {OUTPUT_CSV}")
    else:
        print("Analysis failed or returned empty.")

if __name__ == "__main__":
    main()
