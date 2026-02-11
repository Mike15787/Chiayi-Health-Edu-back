# 建立虛擬環境
python3.10 -m venv venv

# 開啟虛擬環境
source venv/bin/activate


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


debug專用測試程式指令
python tests/auto_tester.py --env auto debug

範例答案進去做測試 15*2
python tests/auto_tester.py golden --env auto

輸出範例答案測試結果
python tests/export_debug_report.py


--------------------
ngrok 啟動方式
ngrok start --all

--------------------
streamlit 啟動方式
用來視覺化整個測試過程
streamlit run review_app.py

vllm 啟動方式(僅限實驗室server)
source vllm-venv/bin/activate

python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-3-4b-it \
    --port 8243 \
    --gpu-memory-utilization 0.7


llama-server啟動方式
目前已經有先寫了一個.sh檔
nano run_server.sh
-------------------------
#!/bin/bash

# 這是執行檔的絕對路徑
/home/braslab/llama.cpp/build/bin/llama-server \
  -m /home/braslab/llama.cpp/models/gemma-3-12b-it-q4_0.gguf \
  --port 8243 \
  -n 512 \
  -ngl 99 \
  -c 4096 \
  --host 0.0.0.0
-----------------------存檔離開 (Ctrl+O -> Enter -> Ctrl+X)
給予執行權限
chmod +x run_server.sh

現在用 PM2 來啟動這個腳本
cd llama.cpp
pm2 start ./run_server.sh --name "llama-gemma"

---------------------------------------------------
#透過pm2管理方式 啟用 前端 後端 ngrok vllm

# 1. 啟動前端 (假設在 frontend 目錄)
cd /path/to/frontend
pm2 start "npm run dev" --name frontend

# 2. 啟動後端 (使用虛擬環境的 python)
cd /path/to/backend
使用vllm (gemma3:4b 無量化)
pm2 start "venv/bin/python run.py --env human --provider vllm" --name "health-backend"
使用llamacpp (gemma3:12b 量化模型)
pm2 start "venv/bin/python run.py --env human --provider llamacpp" --name "health-backend"

# 3. 啟動 ngrok 隧道
pm2 start "ngrok start --all" --name ngrok-tunnels

# 4. 檢查大家是否都 online
pm2 list

用於熱重載
pm2 restart health-backend
pm2 restart my-frontend

ngrok 查看目前通道與連結
curl http://localhost:4040/api/tunnels | python3 -m json.tool | grep -E "name|public_url"


braslab@braslabchiachi:~$ ls
Chiayi-Health-Edu-back  Chiayi-Health-Edu-front  tools  vllm-server
braslab@braslabchiachi:~$ cd vllm-server
braslab@braslabchiachi:~/vllm-server$ source vllm-venv/bin/activate
(vllm-venv) braslab@braslabchiachi:~/vllm-server$ python -m vllm.entrypoints.openai.api_server \
>     --model google/gemma-3-4b-it \
>     --port 8243 \
>     --gpu-memory-utilization 0.75

刪除前端pm2
pm2 delete my-frontend

刪除vllm執行緒
pm2 delete vllm-service


啟動ollama
sudo systemctl start ollama

關閉ollama
sudo systemctl stop ollama
sudo systemctl disable ollama

python tests/voice_replay_tester.py --provider ollama