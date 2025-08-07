import uuid
from datetime import datetime, timezone
from databases import SessionLocal, ChatLog

def parse_and_import_example():
    """
    讀取 example.txt 檔案並將對話內容匯入到資料庫
    """
    # 生成唯一的 session_id 給這次的對話
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
        
        imported_count = 0
        
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
    print("=== 對話資料匯入工具 ===")
    print("這個腳本會將 example.txt 的對話內容匯入到 chatlog 資料表")
    
    # 執行匯入
    parse_and_import_example()
    
    # 驗證匯入結果
    print("\n=== 驗證匯入結果 ===")
    verify_import()
    
    print("\n匯入完成！")