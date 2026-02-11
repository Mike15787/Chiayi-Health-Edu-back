# hydration_and_goal_logic.py
import asyncio
import os
import sys

# 1. 設定路徑與環境變數
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ["LLM_PROVIDER"] = "ollama"

from utils import generate_llm_response

# 請確認你的 Ollama 模型名稱
MODEL_NAME = "gemma3:12b"

# --- 測試用的對話紀錄 (這裡使用一個稍微有瑕疵的案例來測試) ---
# 案例設定：
# 1. 總量 2000cc (O)
# 2. 服藥後一小時 (O)
# 3. 計算: 250cc * 6杯 = 1500cc (X -> 數學錯)
# 4. 時長: 2-3小時 (O)
# 5. 分次喝 (O)
TEST_CONTEXT = """
學員: 藥粉完全溶解的時間,可能約會有五分鐘在溶解的過程當中,水溫可能會略為升高,這是正常的不用擔心
病患: 好的，了解。
學員: 服藥後一小時,也就是晚上的六點,我們可以先準備好2000cc的白開水。
病患: 好的，了解。
學員: 那這2000cc的白開水在兩到三個小時內陸續分次的喝完，我們可以每15分鐘補充一杯250cc的白開水總共喝六杯。
"""

# --- 拆解後的極簡 Prompts ---
# 每個 Prompt 只專注做一件事
CHECK_LIST = [
    {
        "key": "total_amount",
        "title": "檢查 1: 總水量說明",
        "prompt_template": """
    你現在的任務是判斷以下對話紀錄
    學員是否明確告知 總共要準備或喝約 2000cc 的水
        
    [對話紀錄]: {conversation_context}
        
    符合輸出 "1"，不符合或未提及輸出 "0" 不要輸出除了數字以外的文字
""",
    },
    {
        "key": "start_time",
        "title": "檢查 2: 開始喝水時間",
        "prompt_template": """
    你現在的任務是判斷以下對話紀錄
    學員是否明確告知 需在「服藥後一小時」才開始喝水
        
    [對話紀錄]: {conversation_context}
        
    符合輸出 "1"，不符合或未提及輸出 "0" 不要輸出除了數字以外的文字
""",
    },
    {
        "key": "math_logic",
        "title": "檢查 3: 杯數計算邏輯",
        "prompt_template": """
    你現在的任務是判斷以下對話紀錄
    學員是否有說明 要喝 *8* 杯 *150cc* 的水
    如果學員說的 杯數 * cc數 不等於 2000cc 的話 就是錯

    [對話紀錄]: {conversation_context}
        
    符合條件輸出 "1"，不符合輸出 "0" 不要輸出除了數字以外的文字
""",
    },
    {
        "key": "duration",
        "title": "檢查 4: 喝水總時長",
        "prompt_template": """
    你現在的任務是判斷以下對話紀錄
    學員是否明確告知 水要在「兩到三個小時」內喝完
        
    [對話紀錄]: {conversation_context}
        
    符合輸出 "1"，不符合或未提及輸出 "0" 不要輸出除了數字以外的文字
""",
    },
    {
        "key": "split_drinking",
        "title": "檢查 5: 是否分次喝",
        "prompt_template": """
    你現在的任務是判斷以下對話紀錄
    學員是否明確告知 水需要「分次」或「陸續」喝完，而非一次喝完
        
    [對話紀錄]: {conversation_context}
        
    符合輸出 "1"，不符合或未提及輸出 "0" 不要輸出除了數字以外的文字
""",
    },
    {
        "key": "mis",
        "title": "檢查 6: 是否會被150cc水泡藥誤導",
        "prompt_template": """
    你現在的任務是判斷以下對話紀錄
    學員是否明確告知 水需要「分次」或「陸續」喝完，而非一次喝完
        
    [對話紀錄]: {conversation_context}
        
    符合輸出 "1"，不符合或未提及輸出 "0" 不要輸出除了數字以外的文字
""",
    },
]


async def run_split_tests():
    print(f"🚀 開始執行拆分式 Prompt 測試")
    print(f"🤖 使用模型: {MODEL_NAME}")
    print("=" * 60)
    print(f"📄 測試對話摘要: 2000cc | 1小時後 | 250cc*6杯(錯) | 2-3小時 | 分次喝")
    print("-" * 60)

    results = {}

    for check in CHECK_LIST:
        print(f"正在執行: {check['title']} ...", end="\r")

        # 組合 Prompt
        full_prompt = check["prompt_template"].format(conversation_context=TEST_CONTEXT)

        # 呼叫 LLM
        response = await generate_llm_response(full_prompt, model_name=MODEL_NAME)
        result = response.strip()

        results[check["key"]] = result

        # 顯示單項結果
        # 針對這個測試案例，預期 math_logic 應該是 "0"，其他是 "1"
        status_icon = "✅" if result == "1" else "❌"
        if check["key"] == "math_logic":
            # 特殊處理：因為測試案例故意寫錯數學，所以如果回傳 0 其實是模型判斷正確
            status_icon = (
                "✅ (模型成功抓出數學錯誤)" if result == "0" else "⚠️ (模型未抓出錯誤)"
            )

        print(f"{check['title']}: [{result}] {status_icon}")

    print("=" * 60)
    print("📊 總結報告:")

    # 簡易評分邏輯
    score = 0
    total_checks = len(CHECK_LIST)
    for k, v in results.items():
        if k == "math_logic":
            if v == "0":
                score += 1  # 數學錯了要回傳0才算對
        else:
            if v == "1":
                score += 1

    print(f"模型在 5 個檢查點中，成功判斷了 {score} 個。")
    print("註：此測試案例中，'杯數計算邏輯' 應為 0，其餘應為 1。")


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(run_split_tests())
