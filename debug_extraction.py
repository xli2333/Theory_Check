"""
调试工具：测试 AI 提取功能
使用示例文本测试 Gemini API 的响应
"""
import asyncio
import google.generativeai as genai
import json
import os

# 配置
API_KEY = "AIzaSyDaDMWEEp5Dx3FReUyDYcL92aWcNn8jmLI"
MODEL_NAME = 'gemini-2.5-flash'
PROXY_URL = "http://127.0.0.1:7897"

os.environ['HTTP_PROXY'] = PROXY_URL
os.environ['HTTPS_PROXY'] = PROXY_URL
os.environ['NO_PROXY'] = "localhost,127.0.0.1"

genai.configure(api_key=API_KEY)

# 测试文本
TEST_TEXT = """
星巴克中国市场战略案例

背景：
星巴克于1999年进入中国市场。面对本土咖啡文化薄弱、茶饮传统深厚的挑战，星巴克采用了本土化战略。

战略分析：
1. 运用波特五力模型分析中国咖啡市场竞争格局，发现市场集中度低、进入壁垒适中
2. 基于SWOT分析，识别出品牌优势和文化差异的劣势
3. 通过价值链分析，重点优化门店体验和供应链本地化

市场策略：
1. 市场细分：将目标客户定位为25-40岁的城市白领和年轻人
2. 差异化定位：打造"第三空间"概念，而非单纯的咖啡店
3. 产品本土化：推出茶饮系列、月饼等符合中国消费者口味的产品

结果：
截至2023年，星巴克在中国拥有超过6000家门店，成为美国以外最大的市场。
"""

async def test_extraction():
    print("="*60)
    print("AI 提取测试工具")
    print("="*60)

    prompt = f"""
你是一个商业案例分析专家。请仔细分析以下商业案例文本，提取其中的理论框架和知识点。

要求：
1. 提取所有涉及的**理论**（如：波特五力模型、SWOT分析、价值链理论等）
2. 提取所有**知识点**（如：市场细分、品牌定位、供应链管理等）
3. 对于每个理论/知识点，需要提供：
   - category: "理论" 或 "知识点"
   - standard_name: 标准化名称（如"波特五力模型"）
   - context: 在文中出现的上下文（原句，最多100字）

输出格式（严格JSON）：
{{
  "meta": {{
    "title": "案例标题（从文中提取）"
  }},
  "items": [
    {{
      "category": "理论",
      "standard_name": "波特五力模型",
      "context": "该公司运用波特五力分析框架评估行业竞争态势..."
    }},
    {{
      "category": "知识点",
      "standard_name": "市场细分",
      "context": "通过对目标市场进行细分，识别出三类核心客户群体..."
    }}
  ]
}}

案例文本：
{TEST_TEXT}

请输出符合上述格式的JSON（不要包含markdown代码块标记）：
    """

    print("\n📤 发送请求到 Gemini API...")

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        res = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config={"temperature": 0.4},
            request_options={'timeout': 60}
        )

        raw = res.text.strip()
        print(f"\n✅ API 响应成功！")
        print(f"\n📄 原始响应（前1000字符）:\n{'-'*60}")
        print(raw[:1000])
        print(f"{'-'*60}\n")

        # 清理 markdown
        if raw.startswith("```"):
            lines = raw.split('\n')
            raw = '\n'.join(lines[1:])
            if raw.endswith("```"):
                raw = raw[:-3]

        # 提取 JSON
        start = raw.find('{')
        end = raw.rfind('}') + 1
        if start != -1 and end > start:
            raw = raw[start:end]

        # 解析 JSON
        result = json.loads(raw)

        print(f"📊 提取结果:")
        print(f"  - 标题: {result.get('meta', {}).get('title', 'N/A')}")
        print(f"  - 总项目数: {len(result.get('items', []))}")

        theories = [i for i in result.get('items', []) if i.get('category') == '理论']
        points = [i for i in result.get('items', []) if i.get('category') == '知识点']

        print(f"\n🎓 理论 ({len(theories)}):")
        for t in theories:
            print(f"  - {t.get('standard_name')}")
            print(f"    上下文: {t.get('context', '')[:80]}...")

        print(f"\n💡 知识点 ({len(points)}):")
        for p in points:
            print(f"  - {p.get('standard_name')}")
            print(f"    上下文: {p.get('context', '')[:80]}...")

        print(f"\n✅ 测试成功！AI 提取功能正常工作")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_extraction())
