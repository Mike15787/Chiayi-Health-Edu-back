# run.py
import os
import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="啟動 AI Voice Chat API")

    # 新增 --test 參數
    parser.add_argument(
        "--test",
        action="store_true",
        help="啟動測試模式 (使用 chatlog_test.db)，保留資料供他人測試用",
    )

    # 新增 --host 和 --port 參數，方便你調整
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    # --- [新增] LLM 相關參數 ---
    parser.add_argument(
        "--provider",
        type=str,
        choices=["gemini", "ollama"],
        default="ollama",
        help="選擇 LLM 供應商 (預設 ollama)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="指定模型名稱 (例如: gemini-1.5-flash 或 gemma2:9b)。若未指定將使用 config.py 預設值。",
    )

    args = parser.parse_args()

    # 1. 設定資料庫環境
    if args.test:
        os.environ["APP_ENV"] = "test"
        db_name = "chatlog_test.db"
    else:
        os.environ["APP_ENV"] = "dev"
        db_name = "chatlog.db"

    # 2. 設定 LLM 環境變數
    os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        # 如果有指定模型，設定進環境變數，讓 config.py 或 utils.py 讀取
        os.environ["LLM_MODEL_OVERRIDE"] = args.model

    print("==========================================")
    print(f"🚀 啟動模式: {os.environ['APP_ENV'].upper()}")
    print(f"📂 資料庫: {db_name}")
    print(f"🤖 LLM Provider: {args.provider}")
    print(f"🧠 LLM Model: {args.model if args.model else 'Default (from config)'}")
    print("==========================================")

    # 啟動 Uvicorn
    # 注意：這裡使用 factory 模式或直接傳入字串讓 uvicorn 能夠吃到環境變數
    uvicorn.run(
        "main:app", host=args.host, port=args.port, reload=args.reload, log_level="info"
    )


if __name__ == "__main__":
    main()
