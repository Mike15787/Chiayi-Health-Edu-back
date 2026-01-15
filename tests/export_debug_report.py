# tests/export_debug_report.py
import sys
import os
import asyncio
import pandas as pd
from datetime import datetime
from sqlalchemy import desc

# --- 環境路徑設定 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 設定環境變數 (讀取自動測試用的資料庫)
os.environ["APP_ENV"] = "auto"

from databases import SessionLocal, SessionUserMap, Scores, AgentSettings, AnswerLog
from scoring_service_manager import ScoringServiceManager


async def generate_debug_report(output_file="golden_test_report.xlsx"):
    print(f"🚀 開始匯出報表，目標環境: {os.environ.get('APP_ENV')}")

    db = SessionLocal()
    scoring_manager = ScoringServiceManager()

    try:
        # 1. 撈取由 GoldenTester 建立的 Sessions (依照時間倒序，取最近的測試結果)
        # 你可以根據需要調整 limit，例如最近 30 筆 (對應 15 個 Agent * 2 runs)
        target_username = "GoldenTester"
        sessions = (
            db.query(SessionUserMap)
            .filter(SessionUserMap.username == target_username)
            .order_by(desc(SessionUserMap.created_at))
            .limit(50)  # 假設你跑了兩輪 15 個 Agent，約 30 筆，抓 50 筆保險
            .all()
        )

        if not sessions:
            print(
                f"❌ 找不到使用者 '{target_username}' 的資料，請確認是否已執行 auto_tester.py"
            )
            return

        print(f"🔍 找到 {len(sessions)} 筆 Session，正在分析評分細節...")

        all_rows = []
        all_item_ids = set()  # 用來收集所有出現過的評分項目 ID

        # 2. 遍歷每個 Session，還原詳細分數
        for idx, session in enumerate(sessions):
            print(
                f"   [{idx+1}/{len(sessions)}] 處理 Session: {session.agent_code} ({session.session_id})"
            )

            # 取得該 Session 的總分紀錄
            score_record = (
                db.query(Scores).filter(Scores.session_id == session.session_id).first()
            )
            total_score = score_record.total_score if score_record else "0"

            # 呼叫 ScoringService 取得詳細評分結構 (包含邏輯判定與 UI 判定)
            # 這會回傳與前端 API /scoring/details/{id} 一樣的結構
            details = await scoring_manager.get_detailed_scores(
                session.session_id, session.module_id, db
            )

            # 準備這一列的基礎資料
            row_data = {
                "Time": session.created_at.strftime("%Y-%m-%d %H:%M"),
                "Agent": session.agent_code,
                "Total Score": float(total_score),
                "Session ID": session.session_id,
            }

            # 攤平詳細分數結構
            # details 結構: { "CategoryName": { "items": [ {item_id, user_score, weight...} ] } }
            for category_name, cat_data in details.items():
                for item in cat_data["items"]:
                    item_id = item["item_id"]
                    score = item["user_score"]
                    weight = item["weight"]

                    # 收集欄位名稱
                    all_item_ids.add(item_id)

                    # 判斷 O / X / △
                    # 有些項目(如組織效率)可能有小數點
                    if score == weight and weight > 0:
                        mark = "O"
                    elif score == 0:
                        mark = "X"
                    else:
                        mark = f"△ ({score}/{weight})"

                    row_data[item_id] = mark

            # --- B. 【關鍵修改】補抓取被隱藏的 proper_guidance 細項 ---
            # 這些項目在 scoring_logic.py 被濾掉了，但資料庫(AnswerLog)裡有存
            hidden_items = [
                "proper_guidance_s1",
                "proper_guidance_s2",
                "proper_guidance_s3",
                "proper_guidance_s4",
                "proper_guidance_s5",
            ]

            # 直接查 AnswerLog
            raw_logs = (
                db.query(AnswerLog)
                .filter(
                    AnswerLog.session_id == session.session_id,
                    AnswerLog.scoring_item_id.in_(hidden_items),
                )
                .all()
            )

            # 轉成字典方便查找 {item_id: score}
            raw_scores = {log.scoring_item_id: log.score for log in raw_logs}

            for hidden_id in hidden_items:
                all_item_ids.add(hidden_id)
                # 取得分數，預設為 0
                s_val = raw_scores.get(hidden_id, 0)
                # 顯示 O (1分) 或 X (0分)
                row_data[hidden_id] = "O" if s_val == 1 else "X"
                
            all_rows.append(row_data)

        # 3. 轉為 Pandas DataFrame
        df = pd.DataFrame(all_rows)

        # 4. 整理欄位順序
        # 固定欄位放前面
        fixed_cols = ["Time", "Agent", "Total Score", "Session ID"]

        # 動態欄位 (評分項目) 排序，這裡簡單用字母排序，或你可以依照 scoring_criteria 的順序排
        score_cols = [c for c in df.columns if c not in fixed_cols]
        score_cols.sort()

        final_cols = fixed_cols + score_cols
        df = df[final_cols]

        # 5. 輸出 Excel
        # 使用 ExcelWriter 可以進行簡單的格式設定 (例如 Agent 排序)
        df.sort_values(by=["Agent", "Time"], inplace=True)

        df.to_excel(output_file, index=False, engine="openpyxl")

        print(f"\n✅ 報表已生成: {output_file}")
        print(f"   共分析 {len(df)} 筆資料，包含 {len(score_cols)} 個評分項目。")

    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
        import traceback

        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(generate_debug_report())
