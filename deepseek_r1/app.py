from flask import Flask, request, render_template, session, Response, copy_current_request_context
import openai
import os
from datetime import datetime, timedelta
from flask import stream_with_context
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def get_session_messages():

    if "messages" not in session:
        session["messages"] = [SYSTEM_PROMPT]
        session.modified = True
    return session["messages"]

def save_session_messages(messages):
    """安全地保存消息到session"""
    session["messages"] = messages
    session.modified = True

def trim_messages(messages, max_rounds=5):
    """保留最近max_rounds"""
    if len(messages) <= 1:
        return messages
        

    trimmed = [messages[0]]
    

    conversation = messages[1:]  
    pairs = []
    

    i = len(conversation) - 1
    while i > 0 and len(pairs) < max_rounds:
        if conversation[i]["role"] == "assistant" and i > 0 and conversation[i-1]["role"] == "user":
            pairs.append((conversation[i-1], conversation[i]))
            i -= 2
        else:
            i -= 1
    
 
    for user_msg, assistant_msg in reversed(pairs):
        trimmed.extend([user_msg, assistant_msg])
    
    return trimmed

@app.route("/", methods=["GET"])
def index():
    messages = get_session_messages()
    return render_template("index.html", messages=messages)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data:
            return Response("無效的請求數據", status=400)

        user_input = data.get("user_input", "").strip()
        messages = data.get("messages", [])

        if not user_input:
            return Response("請輸入有效內容", status=400)


        all_messages = [SYSTEM_PROMPT] + messages

        def generate():
            try:
                response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=all_messages,
                    stream=True
                )
                
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                        
            except Exception as e:
                yield f"錯誤：{str(e)}"

        return Response(stream_with_context(generate()), mimetype="text/plain")

    except Exception as e:
        return Response(f"處理請求時發生錯誤：{str(e)}", status=500)
@app.route("/reset", methods=["POST"])
def reset():
    try:
        session["messages"] = [SYSTEM_PROMPT]
        session.modified = True
        logger.info("Session reset")
        return render_template("index.html", messages=session["messages"])
    except Exception as e:
        logger.error(f"Error resetting session: {str(e)}")
        return Response(f"重置失敗：{str(e)}", status=500)

if __name__ == "__main__":
    from flask_session import Session
    
    app.config.update(
        SESSION_TYPE='filesystem',
        SESSION_FILE_DIR=os.path.join(app.root_path, 'flask_session'),
        SESSION_PERMANENT=True,
        PERMANENT_SESSION_LIFETIME=timedelta(hours=2)
    )
    
    Session(app)
    

    os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
    

    @app.before_request
    def before_request():
        logger.info(f"Session before request: {dict(session)}")
        
    @app.after_request
    def after_request(response):
        logger.info(f"Session after request: {dict(session)}")
        return response
    
    app.run(host="0.0.0.0", port=5000, debug=True)
