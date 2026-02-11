import asyncio
import os
import sys

# 1. 設定路徑與環境變數
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ["LLM_PROVIDER"] = "ollama"

from utils import generate_llm_response

MODEL_NAME = "gemma3:12b"

# --- 基礎對話 (你提供的泡藥說明) ---
# 這段話只提到了 150cc 泡藥，完全沒提 2000cc 補水
BASE_CONTEXT = """
學員: 接下來跟您說明保可淨要使用的一個時間跟方式那先說明第一包的一個使用時間跟泡製的方式
病患: 好的，請說。

學員: 第一包保可淨使用的時間在檢查前一天也就是2月7日,下午5點的時候,不用第一包,清腸藥保可淨。
病患: 好的，下午五點服用第一包。

學員: 是的,那我們可以先準備一杯裝有150cc白開水的杯子然後把第一包保可淨倒入杯中攪拌約5分鐘等到藥粉完全溶解立即喝完所有的藥水
"""

# --- 測試案例 ---
TEST_CASES = [
    {
        "title": "[測試一] 純干擾測試 (只有泡藥150cc)",
        "description": "對話中只有泡藥的150cc，完全沒提補充水分的2000cc。",
        "context": BASE_CONTEXT,
        "expected": "0",  # 沒提2000cc，也沒提補充水分，應該是0
        "reason": "模型不應將泡藥水的 150cc 誤認為補充水分。",
    },
    {
        "title": "[測試二] 抗干擾測試 (泡藥150cc + 正確補水2000cc)",
        "description": "前面講泡藥150cc，後面正確講述要準備2000cc並分次喝完。",
        "context": BASE_CONTEXT
        + """
        學員: 喝完藥水後，接下來非常重要。
        學員: 請您在接下來兩小時內，準備 2000cc 的白開水。
        學員: 我們可以分成 8 次喝，每次大約 250cc，慢慢把它喝完。
        """,
        "expected": "1",  # 有明確說 250*8=2000，應該是1
        "reason": "模型應忽略前面的 150cc，抓到後面的 2000cc 正確邏輯。",
    },
    {
        "title": "[測試三] 混淆測試 (泡藥150cc + 錯誤補水)",
        "description": "前面講泡藥150cc，後面講補水但總量不夠 (例如只喝 1000cc)。",
        "context": BASE_CONTEXT
        + """
        學員: 喝完藥水後，要多喝水。
        學員: 請您再喝 4 杯水，每杯 250cc 就可以了。
        """,
        # 邏輯: 4*250 = 1000cc (不等於2000)
        # 如果模型把前面的 150cc 亂加進來，或者看到數字就暈，可能會錯
        "expected": "0",
        "reason": "補充水分總量僅 1000cc，不達標。模型不應受 150cc 影響。",
    },
]

# --- 使用極簡 Prompt (針對數學與總量) ---
PROMPT_TEMPLATE = """
    你現在的任務是判斷以下對話紀錄
    學員是否有說明 要喝 *8* 杯 *150cc* 的水
    如果學員說的 杯數 * cc數 不等於 2000cc 的話 就是錯
    
    [對話紀錄]: {conversation_context}
    
    符合條件輸出 "1"，不符合輸出 "0" 不要輸出除了數字以外的文字
"""


async def run_distraction_test():
    print(f"🚀 開始執行抗干擾測試 (150cc 泡藥水陷阱)")
    print(f"🤖 使用模型: {MODEL_NAME}")
    print("=" * 60)

    for case in TEST_CASES:
        print(f"正在執行: {case['title']}")
        print(f"情境: {case['description']}")

        # 填入 Prompt
        full_prompt = PROMPT_TEMPLATE.format(conversation_context=case["context"])

        # 呼叫 LLM
        response = await generate_llm_response(full_prompt, model_name=MODEL_NAME)
        result = response.strip()

        # 驗證
        is_pass = result == case["expected"]
        icon = (
            "✅ Pass"
            if is_pass
            else f"❌ Fail (預期 {case['expected']}, 實際 {result})"
        )

        print(f"👉 模型回應: [{result}]  {icon}")
        if not is_pass:
            print(f"   ⚠️ 失敗原因可能是: {case['reason']}")

        print("-" * 60)

    print("測試結束")


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(run_distraction_test())
