# run.py
import os
import argparse
import uvicorn

# ❌ 注意：這裡不要 import databases，因為還沒設定環境變數


def main():
    parser = argparse.ArgumentParser(description="啟動 AI Voice Chat API")

    parser.add_argument(
        "--env",
        type=str,
        choices=["dev", "human", "auto"],
        default="dev",
        help="選擇執行環境: dev (開發), human (真人測試/Ngrok), auto (自動化測試)",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="[快捷鍵] 等同於 --env auto",
    )

    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    parser.add_argument(
        "--provider",
        type=str,
        choices=["gemini", "ollama", "vllm", "llamacpp"],
        default="ollama",
        help="選擇 LLM 供應商 (預設 ollama)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="指定模型名稱 (例如: gemini-1.5-flash 或 gemma2:9b)。",
    )

    args = parser.parse_args()

    # 1. 設定資料庫環境變數 (必須在 import databases 之前！)
    if args.test:
        os.environ["APP_ENV"] = "auto"
    else:
        os.environ["APP_ENV"] = args.env

    # 2. 設定 LLM 環境變數
    os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["LLM_MODEL_OVERRIDE"] = args.model

    print("==========================================")
    print(f"🚀 啟動模式: {os.environ['APP_ENV'].upper()}")
    print(f"🤖 LLM Provider: {args.provider}")
    print("==========================================")

    # 3. [關鍵修改]：環境變數設定好之後，才 Import databases
    # 這樣 databases.py 才會讀到正確的 APP_ENV
    from databases import init_database

    # 初始化資料庫
    init_database()

    # 啟動 Uvicorn
    # 注意：雖然這裡 main:app 會再次觸發 import，但因為目前 process 的 os.environ 已經設定好了，
    # 所以 uvicorn 載入 main.py -> 載入 databases.py 時，會讀到正確的環境變數。
    uvicorn.run(
        "main:app", host=args.host, port=args.port, reload=args.reload, log_level="info"
    )


if __name__ == "__main__":
    main()
