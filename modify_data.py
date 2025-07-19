import sqlite3

# 連接到 SQLite 資料庫
conn = sqlite3.connect('chatlog.db')
cursor = conn.cursor()

# 刪除 id 為 1 的資料
cursor.execute("DELETE FROM chatlog WHERE id = ?", (2,))

# 提交變更
conn.commit()

# 關閉連線
cursor.close()
conn.close()

print("刪除完成！")