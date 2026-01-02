# export_test_results.py
import asyncio
import os
import pandas as pd
import logging
from datetime import datetime
from sqlalchemy.orm import Session

# 引入你的專案模組
from databases import SessionLocal, SessionUserMap, AgentSettings
from scenarios.colonoscopy_bowklean.scoring_logic import ColonoscopyBowkleanScoringLogic
from scenarios.colonoscopy_bowklean.config import MODULE_ID

# 設定 Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    # 1. 讀取環境變數 (LLM 設定)
    llm_provider = os.getenv("LLM_PROVIDER", "ollama")  # 預設 ollama
    # 嘗試讀取具體模型名稱，若無則標記為 Default
    llm_model = os.getenv("LLM_MODEL_OVERRIDE", "Default (Config)")

    print(f"=== 開始匯出測試結果 ===")
    print(f"目前環境設定 -> Provider: {llm_provider}, Model: {llm_model}")

    # 2. 初始化資料庫與評分邏輯
    db: Session = SessionLocal()
    scoring_logic = ColonoscopyBowkleanScoringLogic()

    try:
        # 3. 撈取該模組的所有 Sessions
        # 你可以依需求過濾，例如只撈取 is_completed=True 的
        sessions = (
            db.query(SessionUserMap)
            .filter(SessionUserMap.module_id == MODULE_ID)
            .order_by(SessionUserMap.created_at.desc())
            .all()
        )

        if not sessions:
            print(f"找不到模組 {MODULE_ID} 的任何 Session 資料。")
            return

        print(f"共找到 {len(sessions)} 筆 Session，開始處理...")

        all_records = []

        for sess in sessions:
            try:
                # 取得 Agent 設定以便在 CSV 顯示案例難度等資訊
                agent_settings = (
                    db.query(AgentSettings)
                    .filter(AgentSettings.agent_code == sess.agent_code)
                    .first()
                )

                agent_complexity = (
                    agent_settings.med_complexity if agent_settings else "未知"
                )
                drug_combination = (
                    agent_settings.drug_combination if agent_settings else "未知"
                )

                # 4. 呼叫 get_detailed_scores
                # 這會回傳一個 Dict: { "檢閱藥歷": {items: [...]}, "醫療面談": {items: [...]}, ... }
                details = await scoring_logic.get_detailed_scores(sess.session_id, db)

                # 5. 攤平資料 (Flattening)
                for category, content in details.items():
                    category_name = content.get("category_name", category)
                    items = content.get("items", [])

                    for item in items:
                        # 處理 relevant_dialogues (List -> String)
                        dialogues_str = " | ".join(item.get("relevant_dialogues", []))

                        record = {
                            # Session 資訊
                            "Session_ID": sess.session_id,
                            "Username": sess.username,
                            "Created_At": sess.created_at,
                            "Agent_Code": sess.agent_code,
                            "Complexity": agent_complexity,
                            "Drug_Combination": drug_combination,
                            # 環境變數 (LLM)
                            "LLM_Provider": llm_provider,
                            "LLM_Model": llm_model,
                            # 評分細項
                            "Category": category_name,
                            "Item_ID": item.get("item_id"),
                            "Item_Name": item.get("item_name"),
                            "Description": item.get("description"),
                            "Scoring_Type": item.get("scoring_type"),
                            # 分數結果
                            "User_Score": item.get("user_score"),
                            "Weight": item.get("weight"),
                            "Is_Passed": (
                                1 if item.get("user_score", 0) > 0 else 0
                            ),  # 簡單判斷是否得分
                            # 相關對話 (方便除錯)
                            "Relevant_Dialogues": dialogues_str,
                        }
                        all_records.append(record)

            except Exception as e:
                logger.error(f"處理 Session {sess.session_id} 時發生錯誤: {e}")
                continue

        # 6. 轉成 DataFrame 並匯出
        if all_records:
            df = pd.DataFrame(all_records)

            # 產生檔名 (含時間戳記)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{MODULE_ID}_{llm_provider}_{timestamp}.csv"

            df.to_csv(
                filename, index=False, encoding="utf-8-sig"
            )  # utf-8-sig 讓 Excel 開啟不亂碼
            print(f"✅ 成功匯出 CSV: {filename}")
            print(f"總計資料列數: {len(df)}")
        else:
            print("沒有產生任何記錄。")

    except Exception as e:
        logger.error(f"匯出過程發生致命錯誤: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
