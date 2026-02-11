import os
import sys
from sqlalchemy import delete

# 1. 設定環境變數，確保讀取到的是 human_test.db (或是你目前卡住的那個資料庫)
# 如果你是開發環境卡住，請改成 "dev"
os.environ["APP_ENV"] = "human"

# 加入路徑以匯入 databases.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from databases import (
    SessionLocal, 
    engine,
    SessionUserMap, 
    ChatLog, 
    AnswerLog, 
    Scores, 
    Summary, 
    ConversationSummary, 
    PrecomputedSessionAnswer, 
    ScoringPromptLog, 
    ScoringAttributionLog, 
    SessionInteractionLog
)

def clean_specific_users():
    db = SessionLocal()
    
    # 指定要刪除的 username
    target_usernames = ["nasker", "user2_Replay"]
    
    print(f"🔍 正在搜尋使用者: {target_usernames} 的所有 Session...")

    try:
        # 1. 找出這些使用者的所有 session_id
        # 我們先查出 ID，這樣才能去刪除其他表格的關聯資料
        sessions_query = db.query(SessionUserMap.session_id).filter(
            SessionUserMap.username.in_(target_usernames)
        )
        
        # 將查詢結果轉為 list
        session_ids = [row[0] for row in sessions_query.all()]
        
        count = len(session_ids)
        if count == 0:
            print("✅ 找不到相關資料，無需刪除。")
            return

        print(f"⚠️ 找到 {count} 筆 Session，準備刪除所有關聯資料...")
        
        # 定義所有有關聯 session_id 的表格模型
        # 注意：SessionUserMap 必須最後刪除，因為它是主表
        tables_to_clean = [
            ChatLog,
            AnswerLog,
            Scores,
            Summary,
            ConversationSummary,
            PrecomputedSessionAnswer,
            ScoringPromptLog,
            ScoringAttributionLog,
            SessionInteractionLog
        ]

        # 為了避免 SQLite 限制 (too many SQL variables)，我們分批處理
        batch_size = 500
        total_deleted = 0

        for i in range(0, count, batch_size):
            batch_ids = session_ids[i : i + batch_size]
            print(f"   正在處理批次 {i} ~ {i+len(batch_ids)} ...")

            # A. 刪除關聯表格資料
            for table_model in tables_to_clean:
                stmt = delete(table_model).where(table_model.session_id.in_(batch_ids))
                db.execute(stmt)
            
            # B. 刪除 SessionUserMap (主表)
            stmt_main = delete(SessionUserMap).where(SessionUserMap.session_id.in_(batch_ids))
            db.execute(stmt_main)
            
            db.commit()
            total_deleted += len(batch_ids)

        print(f"✅ 成功刪除 {total_deleted} 筆 Session 及其所有關聯資料。")

        # 2. 執行 VACUUM (關鍵步驟)
        # SQLite 刪除資料後不會自動釋放硬碟空間，必須執行 VACUUM 才會變小
        print("🧹 正在執行資料庫重組 (VACUUM)... 這可能需要幾秒鐘...")
        db.execute(text("VACUUM"))  # 使用 text() 包裝 raw sql
        print("✨ 資料庫瘦身完成！")

    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    from sqlalchemy import text # 補 import
    
    # 再次確認
    confirm = input("⚠️ 此操作將永久刪除 'nasker' 和 'user2_Replay' 的所有資料。\n確認請輸入 'yes': ")
    if confirm.lower() == "yes":
        clean_specific_users()
    else:
        print("取消操作。")