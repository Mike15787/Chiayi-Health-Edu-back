import uuid
import os
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from databases import SessionLocal, ChatLog, AgentSettings, Base, sync_db_schema 
from agentset import data_list  # 引入最新的病例資料

# 定義要更新的目標資料庫檔案列表
TARGET_DBS = ["auto_test.db", "human_test.db"]

def sync_agents_to_all_dbs():
    """
    將 agentset.py 中的 data_list 同步更新到所有目標資料庫。
    邏輯：如果有對應 agent_code 則更新欄位，沒有則新增。
    """
    print(f"🚀 開始同步 Agent 資料到以下資料庫: {TARGET_DBS}")
    
    for db_file in TARGET_DBS:
        if not os.path.exists(db_file):
            print(f"⚠️  警告: 找不到檔案 {db_file}，將自動建立並初始化表格。")
        
        # 動態建立連線
        db_url = f"sqlite:///{db_file}"
        engine = create_engine(db_url, echo=False)
        
        print(f"\n📂 正在處理資料庫: 【{db_file}】")

        # 2. [關鍵修改] 針對當前的資料庫引擎，執行結構修補
        try:
            print("   🔧 檢查並修補資料庫欄位...")
            sync_db_schema(engine)  # <--- 這行會自動把缺少的欄位補進去
        except Exception as e:
            print(f"   ⚠️ 修補結構時發生警告 (通常可忽略): {e}")

        # 確保表格存在
        Base.metadata.create_all(engine)
        
        SessionTemp = sessionmaker(bind=engine)
        db = SessionTemp()
        
        try:
            updated_count = 0
            inserted_count = 0
            
            for item in data_list:
                # 資料預處理
                if "不知道檢查型態" in item.get("special_status", ""):
                    item["check_type"] = "不知道"

                # 查詢該 Agent 是否已存在
                existing_agent = db.query(AgentSettings).filter(AgentSettings.agent_code == item["agent_code"]).first()
                
                if existing_agent:
                    # --- 更新模式 (Update) ---
                    has_changes = False
                    for key, value in item.items():
                        if hasattr(existing_agent, key) and getattr(existing_agent, key) != value:
                            setattr(existing_agent, key, value)
                            has_changes = True
                    
                    if has_changes:
                        updated_count += 1
                else:
                    # --- 新增模式 (Insert) ---
                    new_agent = AgentSettings(**item)
                    db.add(new_agent)
                    inserted_count += 1
                    print(f"   ➕ 新增: {item['agent_code']}")
            
            db.commit()
            print(f"   ✅ 完成！新增: {inserted_count} 筆, 更新: {updated_count} 筆")
            
        except Exception as e:
            print(f"   ❌ 處理資料錯誤: {e}")
            db.rollback()
        finally:
            db.close()


def parse_and_import():
    session_id = str(uuid.uuid4())
    agent_code = "A5"
    
    print(f"開始匯入對話，Session ID: {session_id}")
    print(f"Agent Code: {agent_code}")
    
    # 開啟資料庫連線
    db = SessionLocal()
    
    try:
        # 讀取檔案
        with open('example.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        imported_count = 0 #單純計數多少筆對話
        
        for line_num, line in enumerate(lines, 1):
            # 移除前後空白字元
            line = line.strip()
            
            # 跳過空行
            if not line:
                continue
            
            # 判斷角色並提取內容
            role = None
            text = None
            
            if line.startswith("User(醫生):"):
                role = "user"
                text = line.replace("User(醫生):", "").strip()
            elif line.startswith("Agent(病人):"):
                role = "patient"  # 或者你可能想用 "assistant" 
                text = line.replace("Agent(病人):", "").strip()
            else:
                print(f"第 {line_num} 行格式不符，跳過: {line}")
                continue
            
            # 確保有內容才插入
            if text:
                # 創建 ChatLog 記錄
                chat_log = ChatLog(
                    session_id=session_id,
                    agent_code=agent_code,
                    role=role,
                    text=text,
                    time=datetime.now(timezone.utc)
                )
                
                db.add(chat_log)
                imported_count += 1
                
                print(f"第 {line_num} 行 [{role}]: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # 提交所有變更
        db.commit()
        print(f"\n成功匯入 {imported_count} 筆對話記錄到資料庫")
        print(f"Session ID: {session_id}")
    except FileNotFoundError:
        print("錯誤: 找不到 example.txt 檔案，請確認檔案路徑是否正確")
    except Exception as e:
        print(f"匯入過程中發生錯誤: {e}")
        db.rollback()
    finally:
        db.close()

def verify_import(session_id=None):
    """
    驗證匯入的資料
    """
    db = SessionLocal()
    try:
        query = db.query(ChatLog)
        if session_id:
            query = query.filter(ChatLog.session_id == session_id)
        
        # 取得最新的記錄
        records = query.order_by(ChatLog.time.desc()).limit(10).all()
        
        print("\n=== 最新的 10 筆記錄 ===")
        for record in records:
            print(f"ID: {record.id}, Session: {record.session_id[:8]}..., "
                  f"Agent: {record.agent_code}, Role: {record.role}, "
                  f"Text: {record.text[:30]}{'...' if len(record.text) > 30 else ''}")
        
    except Exception as e:
        print(f"驗證資料時發生錯誤: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="資料庫維護工具")
    parser.add_argument("--sync-agents", action="store_true", help="同步 agentset.py 的資料到所有資料庫")
    parser.add_argument("--import-chat", action="store_true", help="匯入 example.txt 的對話紀錄")
    
    args = parser.parse_args()

    # 如果沒有參數，顯示選單
    if not args.sync_agents and not args.import_chat:
        print("=== 資料庫維護工具 ===")
        print("1. 同步 Agent 資料 (chatlog.db, auto_test.db, human_test.db)")
        print("2. 匯入對話記錄 (example.txt -> chatlog.db)")
        choice = input("請選擇功能 (1/2): ")
        
        if choice == "1":
            sync_agents_to_all_dbs()
        elif choice == "2":
            parse_and_import()
            verify_import()
        else:
            print("無效的選擇")
    else:
        # 命令列模式
        if args.sync_agents:
            sync_agents_to_all_dbs()
        
        if args.import_chat:
            parse_and_import()
            verify_import()