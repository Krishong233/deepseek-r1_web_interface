from flask import Flask, request, render_template, session, Response, copy_current_request_context
import openai
import os
from datetime import datetime, timedelta
from flask import stream_with_context

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

# 環境變數設定
api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("請設置 DEEPSEEK_API_KEY 環境變數")

# 初始化 DeepSeek 客戶端
client = openai.OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1"
)

# 系統提示詞
SYSTEM_PROMPT = {
    "role": "system", 
    "content": "You are DeepSeek-R1, a highly intelligent assistant. Respond using clear, Traditional Chinese."
}

# 初始化 session 中的 messages
def init_session_messages():
    if "messages" not in session:
        session["messages"] = [SYSTEM_PROMPT]
        session["last_reset"] = datetime.now().isoformat()

# 消息修剪邏輯
def trim_messages(messages, max_rounds=5):
    """保留最近 max_rounds 輪有效對話"""
    valid_messages = [messages[0]]  # 保留系統提示
    
    # 從新到舊遍歷消息
    user_turn = False  # 標識下一輪應出現的角色
    for msg in reversed(messages[1:]):
        if msg["role"] == ("user" if user_turn else "assistant"):
            valid_messages.insert(1, msg)  # 插入到系統提示後
            user_turn = not user_turn
            if len(valid_messages) >= 1 + 2 * max_rounds:
                break
    
    return valid_messages

@app.route("/", methods=["GET"])
def index():
    init_session_messages()
    return render_template("index.html", messages=session["messages"])

@app.route("/chat", methods=["POST"])
def chat():
    init_session_messages()

    user_input = request.form.get("user_input", "").strip()
    if not user_input:
        return Response("請輸入有效內容", mimetype="text/plain")

    messages = session.get("messages", [])
    
    # 檢查最後一條消息
    if messages and messages[-1]["role"] == "user":
        return Response("等待助理回覆中...", mimetype="text/plain")
        
    # 添加用戶消息
    messages.append({"role": "user", "content": user_input})
    
    # 修剪消息歷史（在添加新消息後進行）
    if len(messages) > 11:  # 5輪對話 + 系統提示 = 11條消息
        messages = trim_messages(messages, max_rounds=5)

    @copy_current_request_context
    def generate():
        try:
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=messages,
                stream=True
            )
            
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
                    
            # 只有在成功生成完整響應後才更新session
            messages.append({"role": "assistant", "content": full_response})
            session["messages"] = messages
            session.modified = True
            
        except Exception as e:
            messages.pop() # 移除失敗的用戶消息
            session["messages"] = messages
            session.modified = True
            yield f"錯誤：{str(e)}"
            
    return Response(stream_with_context(generate()), mimetype="text/plain")

@app.route("/reset", methods=["POST"])
def reset():
    try:
        session["messages"] = [SYSTEM_PROMPT]
        session["last_reset"] = datetime.now().isoformat()
        session.modified = True
        return render_template("index.html", messages=session["messages"])
    except Exception as e:
        return Response(f"重置失敗：{str(e)}", status=500)

if __name__ == "__main__":
    # 添加 session 配置
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)
    
    app.run(host="0.0.0.0", port=5000, debug=True)
