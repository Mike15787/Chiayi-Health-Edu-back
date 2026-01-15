執行步驟 現在要先設定環境變數 APIkey
$env:GEMINI_API_KEY="api key"
再來設置gemini 或 ollama
$env:LLM_PROVIDER="gemini"
$env:LLM_PROVIDER="ollama"

由於有三個測試環境 
初始化開發資料庫 (chatlog.db)
python agentset.py
---------------------------------------
初始化真人測試資料庫 (human_test.db)
set APP_ENV=human
python agentset.py
$env:APP_ENV="human"; python agentset.py

----------------------------------------
初始化自動測試資料庫 (auto_test.db)
set APP_ENV=auto
python agentset.py

# Windows (PowerShell)
$env:APP_ENV="auto"; python agentset.py


--------------------------------

一般開發測試 (Backend/Frontend 串接)
python run.py
# 或
python run.py --env dev
# 使用 chatlog.db

----------------------------------
開放給藥師測試 (Ngrok)
python run.py --env human
# 使用 human_test.db (此時 ngrok 連進來的對話都會存在這)
----------------------------------
執行自動化評分測試
python tests/auto_tester.py golden
# 或
python tests/auto_tester.py sim
# 使用 auto_test.db

