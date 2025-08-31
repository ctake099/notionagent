import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mcp import stdio_client, StdioServerParameters
from strands import Agent
from strands.tools.mcp import MCPClient

# .envファイルから環境変数を読み込む（サーバー起動時に一度だけ実行）
load_dotenv()

# FastAPIアプリケーションを初期化
app = FastAPI(
    title="Notion Agent API",
    description="Notion上の情報を元に質問に答えるAIエージェントAPI"
)

# .envからモデルIDを読み込む
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")

# --- APIの入力と出力の形式を定義 ---
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str

# --- APIエンドポイントの作成 ---
@app.post("/ask", response_model=AnswerResponse)
async def ask_agent(request: QuestionRequest):
    """
    エージェントに質問を投げ、回答を取得するエンドポイント
    """
    user_question = request.question
    print(f"受け取った質問: {user_question}")

    try:
        notion_mcp_client = MCPClient(
            lambda: stdio_client(
                StdioServerParameters(
                    command="npx",
                    args=["-y", "mcp-remote", "https://mcp.notion.com/mcp"]
                )
            )
        )

        print("エージェントを起動しています...")

        with notion_mcp_client:
            tools = notion_mcp_client.list_tools_sync()
            print(f"利用可能なツール: {[tool.tool_name for tool in tools]}")

            agent = Agent(tools=tools, model=BEDROCK_MODEL_ID)

            print("AIエージェントが自律的に思考と行動を開始します...")
            
            final_response = agent(user_question)
            
            print("\n--- エージェントからの最終的な回答 ---")
            print(final_response)

        # 成功した結果を返す
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # 修正点：AgentResultオブジェクトをstr()で文字列に変換する
        return AnswerResponse(question=user_question, answer=str(final_response))
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Notion Agent APIへようこそ！ /docs にアクセスしてAPI仕様を確認してください。"}