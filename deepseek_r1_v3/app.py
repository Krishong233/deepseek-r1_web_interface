from flask import Flask, request, render_template, Response, stream_with_context, session, redirect, url_for, jsonify
import openai
import os
import logging
import json
from functools import wraps
import random
import redis
import re
import time
import pandas as pd
import sqlite3
import datetime
import requests

# Initialize Redis for rate limiting and IP blocking
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

# Environment variable setup
api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("請設置 DEEPSEEK_API_KEY 環境變數")

# Initialize DeepSeek client
client = openai.OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1"
)

# Database setup for spreadsheet quiz
DATABASE = 'quiz_progress.db'

def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS quiz_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            user_answer TEXT,
            correct_answer TEXT,
            correct_flag INTEGER,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Define questions and answers for authentication
questions = [
    {"q": "我的 classno 是什麼？", "a": "23"},
    {"q": "我的中文全名是什麼？", "a": "洪麒翔"},
    {"q": "我的生日是？（dd/mm）", "a": "07/08"}
]

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# --------------------- Authentication Routes ---------------------
@app.route("/get_question", methods=["GET"])
def get_question():
    if session.get("blocked"):
        return "您的 IP 已被封鎖", 403
    question = random.choice(questions)
    session["current_question"] = question["q"]
    session["correct_answer"] = question["a"]
    session["attempts"] = 0  # Reset attempts
    return question["q"]

@app.route("/validate_answer", methods=["POST"])
def validate_answer():
    ip = request.remote_addr
    attempts_key = f"attempts:{ip}"
    block_key = f"blocked:{ip}"
    if redis_client.get(block_key):
        return jsonify({"success": False, "blocked": True})
    data = request.get_json()
    answer = data.get("answer")
    correct_answer = session.get("correct_answer")
    if answer == correct_answer:
        session.permanent = True
        session["authenticated"] = True
        redis_client.delete(attempts_key)
        return jsonify({"success": True})
    else:
        attempts = int(redis_client.get(attempts_key) or 0) + 1
        redis_client.setex(attempts_key, 3600, attempts)  # 1 小時內有效
        if attempts >= 4:
            redis_client.setex(block_key, 86400, 1)  # 封鎖 24 小時
            return jsonify({"success": False, "blocked": True})
        return jsonify({"success": False, "attempts": attempts})

@app.route("/blocked_ips", methods=["GET"])
def get_blocked_ips():
    blocked_keys = redis_client.keys("blocked:*")
    blocked_ips = [key.decode().split(":")[1] for key in blocked_keys]
    return jsonify({"blocked_ips": blocked_ips})

@app.route("/unblock_ip/<ip>", methods=["POST"])
def unblock_ip(ip):
    block_key = f"blocked:{ip}"
    attempts_key = f"attempts:{ip}"
    redis_client.delete(block_key)
    redis_client.delete(attempts_key)
    return jsonify({"success": True, "message": f"IP {ip} 已解鎖"})

@app.route("/unblock_all", methods=["POST"])
def unblock_all():
    blocked_keys = redis_client.keys("blocked:*")
    for key in blocked_keys:
        redis_client.delete(key)
    return jsonify({"success": True, "message": "所有封鎖的 IP 已解除"})

# --------------------- Chatroom Routes ---------------------
@app.route("/chatroom", methods=["GET"])
@login_required
def chatroom():
    return render_template("chatroom.html")

@app.route("/chat", methods=["POST"])
@login_required
def chat():
    try:
        data = request.get_json()
        if not data or "messages" not in data:
            return Response("無效的請求數據", status=400)

        messages = data["messages"]
        if not messages or not any(msg["role"] == "user" for msg in messages):
            return Response("請輸入有效內容", status=400)

        logger.info(f"發送的訊息序列：{messages}")

        def generate():
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
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
@login_required
def reset():
    return render_template("chatroom.html")

# --------------------- SQL Quiz Routes ---------------------
def count_subqueries(sql):
    return len(re.findall(r'\(\s*SELECT', sql, re.IGNORECASE))

# Define fallback questions
sql_questions = [
    {"q": "Write a SQL query to select all employees from the employees table.", "a": "SELECT * FROM employees;"},
    {"q": "Write a SQL query to find the average salary of employees in the employees table.", "a": "SELECT AVG(salary) FROM employees;"},
    {"q": "Write a SQL query to count the number of customers registered after January 1, 2020.", "a": "SELECT COUNT(*) FROM customers WHERE registration_date > '2020-01-01';"}
]

@app.route("/sql-test", methods=["GET"])
def sql_test():
    return render_template("sql-test.html")

@app.route("/get_sql_question", methods=["GET"])
def get_sql_question():
    random_seed = random.randint(1, 1000)
    timestamp = int(time.time())
    prompt = f"""
Generate a SQL quiz question and its correct SQL answer based on the following table schemas.
The quiz question and answer must be in English.
Restrictions:
1. Only use the following SQL keywords, operators, and functions: FALSE, TRUE, AND, NOT, OR, ABS, AVG, INT, MAX, MIN, SUM, COUNT, ASC, AT, CHAR, CHR, CHAR_LENGTH, LEN, LOWER, TRIM, SPACE, SUBSTRING, SUBSTR, MID, UPPER, VALUE, VAL, DATE, DAY, MONTH, YEAR, ADD, ALL, ALTER, ANY, AS, BETWEEN, BY, CREATE, DELETE, DESC, DISTINCT, DROP, EXISTS, FROM, GROUP, HAVING, IN, INDEX, INNER, JOIN, INSERT, INTEGER, INTERSECT, INTO, LEFT, LIKE, MINUS, NULL, RIGHT, FULL, ON, ORDER, SELECT, SET, TABLE, TO, UNION, UNIQUE, UPDATE, VALUES, VIEW, WHERE, +, -, *, /, >, <, =, >=, <=, <>, %, _, '.
2. The SQL statement (correct answer) may contain at most one sub-query (no more than one in whole answer), 1/3 difficulty level.
3. Output your result as a JSON object with exactly two keys: "question" and "answer". Do not include any additional text.
4. To ensure variety, generate a unique question each time, using this random seed: {random_seed} and timestamp: {timestamp}.
Provided Table Schemas:
-- employees table
employee_id   INTEGER    -- Primary Key
name          CHAR(100)
salary        INTEGER
department_id INTEGER    -- Foreign Key

-- customers table
customer_id        INTEGER  -- Primary Key
name               CHAR(100)
registration_date  DATE

-- products table
product_id    INTEGER   -- Primary Key
product_name  CHAR(100)
price         INTEGER

-- orders table
order_id      INTEGER   -- Primary Key
customer_id   INTEGER   -- Foreign Key
total_amount  INTEGER

-- departments table
department_id    INTEGER   -- Primary Key
department_name  CHAR(100)
"""
    messages = [
        {"role": "system", "content": "You are a SQL quiz generator."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            presence_penalty=1,
            messages=messages,
            max_tokens=1000,
            temperature=2,
        )
        generated_text = response.choices[0].message.content.strip()
        
        logger.info(f"Raw API response: {generated_text}")
        
        json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)
        else:
            raise ValueError("No JSON found in API response")
        
        result = json.loads(json_text)
        question_text = result.get("question", "")
        answer_text = result.get("answer", "")
        if not question_text or not answer_text:
            raise ValueError("Missing question or answer in API response")
    except Exception as e:
        logger.error(f"Error generating SQL quiz question: {e}")
        chosen = random.choice(sql_questions)
        question_text = chosen["q"]
        answer_text = chosen["a"]

    session["sql_correct_answer"] = answer_text.strip().lower()
    
    table_schema = """
Provided Table Schemas:
-- employees table
employee_id   INTEGER    -- Primary Key
name          CHAR(100)
salary        INTEGER
department_id INTEGER    -- Foreign Key

-- customers table
customer_id        INTEGER  -- Primary Key
name               CHAR(100)
registration_date  DATE

-- products table
product_id    INTEGER   -- Primary Key
product_name  CHAR(100)
price         INTEGER

-- orders table
order_id      INTEGER   -- Primary Key
customer_id   INTEGER   -- Foreign Key
total_amount  INTEGER

-- departments table
department_id    INTEGER   -- Primary Key
department_name  CHAR(100)
"""
    full_question = question_text
    response = jsonify({"question": full_question})
    response.headers["Cache-Control"] = "no-store"
    return response

@app.route("/validate_sql_answer", methods=["POST"])
def validate_sql_answer():
    data = request.get_json()
    answer = data.get("answer", "")
    
    expected = session.get("sql_correct_answer", "").strip().lower()
    if answer.strip().lower() == expected:
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "message": f"\n{expected}"})

# --------------------- Spreadsheet Quiz Routes ---------------------
@app.route("/spreadsheet-test", methods=['GET', 'POST'])
def spreadsheet_test():
    if request.method == 'POST':
        user_answer = request.form.get('answer').strip()
        if 'current_question' not in session:
            return redirect(url_for('spreadsheet_test'))
        
        correct_answer = session['current_question']['correct_answer']
        question_text = session['current_question']['question']

        is_correct = (user_answer.lower().replace(" ", "") == 
                     correct_answer.lower().replace(" ", ""))

        # Save to database
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute('''INSERT INTO quiz_results 
                    (question, user_answer, correct_answer, correct_flag, timestamp) 
                    VALUES (?, ?, ?, ?, ?)''',
                 (question_text, user_answer, correct_answer, 
                  int(is_correct), datetime.datetime.now().isoformat()))
        conn.commit()
        conn.close()

        result = "Correct!" if is_correct else f"Incorrect. The right answer was: {correct_answer}"
        q, table_html = generate_question()
        return render_template("spreadsheet-test.html", question=q, table_html=table_html, result=result)
    else:
        q, table_html = generate_question()
        return render_template("spreadsheet-test.html", question=q, table_html=table_html, result="")

@app.route('/spreadsheet-progress')
def spreadsheet_progress():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''SELECT question, user_answer, correct_answer, correct_flag, timestamp 
                FROM quiz_results ORDER BY id DESC LIMIT 50''')
    records = c.fetchall()
    conn.close()
    return render_template("progress.html", records=records)

def generate_question():
    try:
        question_data = generate_question_with_deepseek()
        
        if not isinstance(question_data.get("table_data"), dict):
            raise ValueError("Invalid table data format")
            
        columns = [str(col) for col in question_data["table_data"]["columns"]]
        rows = [
            [str(cell) for cell in row] 
            for row in question_data["table_data"]["rows"]
        ]
        
        df = pd.DataFrame(rows, columns=columns)
        
        question_lower = question_data["question"].lower()
        for col_letter in ["d", "e", "f"]:
            if f"column {col_letter}" in question_lower and chr(64 + ord(col_letter)) not in df.columns:
                df[chr(64 + ord(col_letter))] = ""
        
        table_html = build_excel_style_table(df)
        
        session['current_question'] = {
            'question': question_data["question"],
            'table_html': table_html,
            'correct_answer': question_data["formula"],
            'data_columns': list(df.columns)
        }
        
        return question_data["question"], table_html
        
    except Exception as e:
        logger.error(f"Question Generation Error: {str(e)}")
        return (
            "Calculate total price in Column D (Quantity B * Unit Price C)",
            build_excel_style_table(pd.DataFrame({
                "A": ["Product", "Widget", "Gadget"],
                "B": ["Quantity", 5, 3],
                "C": ["Unit Price", 12.99, 24.50],
                "D": ["Total", "", ""]
            }))
        )

def build_excel_style_table(df):
    def escape_html(cell):
        return str(cell).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    col_headers = "".join(
        f"<th class='header-cell'>{chr(65 + i)}</th>"
        for i in range(len(df.columns))
    )
    
    rows = []
    for i in range(len(df)):
        row_cells = "".join(
            f"<td>{escape_html(df.iloc[i][col])}</td>"
            for col in df.columns
        )
        rows.append(f"<tr><td class='header-cell'>{i + 1}</td>{row_cells}</tr>")
    
    return f"""
    <style>
        .excel-table {{ border-collapse: collapse; margin: 20px 0; width: 100%; }}
        .excel-table th, .excel-table td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
            min-width: 80px;
        }}
        .header-cell {{
            background-color: #f2f2f2;
            font-weight: bold;
            color: #333;
        }}
    </style>
    <table class='excel-table'>
        <tr><th class='header-cell'></th>{col_headers}</tr>
        {"".join(rows)}
    </table>
    """

def generate_question_with_deepseek():
    prompt = """Generate a challenging spreadsheet practice question using ONLY the following functions:
- Logical: TRUE, FALSE, AND, NOT, OR, IF, ISBLANK
- Math: ABS, INT, RAND, SQRT, ROUND
- Aggregation: AVERAGE, MAX, MIN, SUM, SUMIF
- Counting: COUNT, COUNTA, COUNTBLANK, COUNTIF
- Text: LOWER, UPPER, PROPER, TRIM, LEN, LEFT, RIGHT, MID, CONCATENATE (&), TEXT, VALUE, CHAR, FIND
- Lookup: VLOOKUP, RANK

STRICT REQUIREMENTS:
1. Question must REQUIRE using at least one of these functions (but don't mention function names)
2. Make the question challenging with multiple conditions or steps
3. Provide rich table data with 5-8 columns and 6-10 rows
4. Question should specify exactly which columns and rows to use(can be multiples)
5. Include the correct formula answer
6. Return ONLY JSON in this exact format:
{
    "question": "Clear English question text specifying columns",
    "table_data": {
        "columns": ["Column1", "Column2", ...],
        "rows": [
            ["Value1", "Value2", ...],
            ...
        ]
    },
    "formula": "=CORRECT_FORMULA",
    "function_type": "FUNCTION_CATEGORY"
}"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=3000
        )
        
        message_content = response.choices[0].message.content
        
        if '```json' in message_content:
            message_content = message_content.split('```json')[1].split('```')[0]
        
        api_data = json.loads(message_content.strip())
        
        required_keys = {"question", "table_data", "formula"}
        if not all(key in api_data for key in required_keys):
            raise ValueError("Incomplete API response")
            
        if len(api_data["table_data"]["rows"]) < 5:
            raise ValueError("Insufficient table rows for complex question")
            
        return api_data
        
    except Exception as e:
        logger.error(f"DeepSeek API Error: {str(e)}")
        return generate_local_question()

def generate_local_question():
    examples = [
        {
            "question": "Calculate bonus eligibility (column D): Return 'Bonus' if Salary (column C) > 5000 and Department (column B) is 'Sales', otherwise 'Standard'",
            "table_data": {
                "columns": ["Employee", "Department", "Salary"],
                "rows": [
                    ["E101", "Sales", 5500],
                    ["E102", "HR", 4800],
                    ["E103", "IT", 5200],
                    ["E104", "Sales", 6100],
                    ["E105", "Sales", 4900]
                ]
            },
            "formula": "=IF(AND(C2>5000,B2=\"Sales\"),\"Bonus\",\"Standard\")",
            "function_type": "IF"
        },
        {
            "question": "Calculate total price (column D) as Quantity (column B) multiplied by Unit Price (column C)",
            "table_data": {
                "columns": ["Product", "Quantity", "Unit Price"],
                "rows": [
                    ["Widget", 5, 12.99],
                    ["Gadget", 3, 24.50],
                    ["Tool", 10, 8.75],
                    ["Part", 2, 15.00],
                    ["Supply", 7, 3.25]
                ]
            },
            "formula": "=B2*C2",
            "function_type": "MULTIPLICATION"
        }
    ]
    return random.choice(examples)

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=True)
