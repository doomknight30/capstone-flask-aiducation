from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import random
import os
import requests
import json
import  google.generativeai as genai
from flask_dance.contrib.google import make_google_blueprint, google
from flask import Flask
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "fallback_secret")
group_rooms = {}

# Google Generative AI configuration
genai.configure(api_key=os.environ.get("GOOGLE_GENAI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

def gemini_complete(messages, model_name="gemini-2.5-flash"):
    """Call Google Gemini API and return the assistant message text."""
    try:
        # Convert OpenAI-style messages to Gemini format
        prompt = ""
        for message in messages:
            if message["role"] == "system":
                prompt += f"System: {message['content']}\n"
            elif message["role"] == "user":
                prompt += f"User: {message['content']}\n"
            elif message["role"] == "assistant":
                prompt += f"Assistant: {message['content']}\n"
        
        # Generate content using Gemini
        response = model.generate_content(prompt)
        result = response.text.strip()
        print(f"Gemini API response: {result[:200]}...")  # Log first 200 chars for debugging
        return result
        
    except Exception as e:
        print(f"Gemini API error: {e}")
        # Fallback to simple response
        if messages and len(messages) > 0:
            last_user_message = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), "")
            return f"I'm having trouble connecting to the AI service. Please try again. (Request: {last_user_message[:50]}...)"
        return "I'm having trouble connecting to the AI service. Please try again."

# File upload config
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_uploaded_file(file_storage):
    """Just search and extract a random but logically complete plain text from an uploaded file (PDF, DOCX, TXT). Returns text or raises Exception."""
    filename = secure_filename(file_storage.filename)
    ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""

    if ext == "txt":
        # Read as plain text
        raw = file_storage.read()
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return raw.decode("latin-1", errors="ignore")

    if ext == "pdf":
        try:
            import PyPDF2
        except ImportError:
            raise Exception("PDF support requires PyPDF2. Install with: pip install PyPDF2")
        try:
            reader = PyPDF2.PdfReader(file_storage)
            pages_text = []
            for page in reader.pages:
                pages_text.append(page.extract_text() or "")
            return "\n".join(pages_text)
        except Exception as e:
            raise Exception(f"Failed to read PDF: {str(e)}")

    if ext == "docx":
        try:
            import docx
        except ImportError:
            raise Exception("DOCX support requires python-docx. Install with: pip install python-docx")
        try:
            document = docx.Document(file_storage)
            return "\n".join(p.text for p in document.paragraphs)
        except Exception as e:
            raise Exception(f"Failed to read DOCX: {str(e)}")

    raise Exception("Unsupported file type")

# Database connection
def connect_db():
    return mysql.connector.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        user=os.environ.get("DB_USER", "root"),
        password=os.environ.get("DB_PASSWORD", ""),
        database=os.environ.get("DB_NAME", "capstone")
    )

def admin_required(f):
    @wraps(f)  # This preserves the function attributes
    def wrapper(*args, **kwargs):
        if "username" not in session or session.get("role") != "admin":
            return redirect(url_for("home"))
        return f(*args, **kwargs)
    return wrapper

@app.route("/", methods=["GET", "POST"])
def home():
    error = None
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        mode = request.form.get("mode")  # "registered" or "guest"
        
        if mode == "registered":
            conn = connect_db()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
            conn.close()
            
            if user and check_password_hash(user['password'], password):
                session["username"] = username
                session["guest"] = False
                session["score"] = 0
                session["question_count"] = 0
                session["role"] = user.get("role")
                
                return redirect(url_for("homepage"))
            else:
                error = "Invalid username or password"

        elif mode == "guest":
            session["username"] = username
            session["guest"] = True
            session["score"] = 0
            session["question_count"] = 0
            return redirect(url_for("homepage"))

    return render_template("login.html", error=error)

@app.route("/quiz_customization", methods=["GET"])
def quiz_customization():
    if "username" not in session:
        return redirect(url_for("home"))
    return render_template("quiz-customization.html")

@app.route("/start_quiz_custom", methods=["POST"])
def start_quiz_custom():
    if "username" not in session:
        return redirect(url_for("home"))

    # Get customization info from form
    quiz_format = request.form.get("format")
    num_items = int(request.form.get("num_items", 5))
    timed = request.form.get("timed")
    time_limit = request.form.get("time_limit")
    if time_limit is not None and time_limit != "":
        time_limit = int(time_limit)
    else:
        time_limit = 0
    session["time_limit"] = time_limit

    session["quiz_format"] = quiz_format
    session["num_items"] = num_items
    session["timed"] = timed
    session["time_limit"] = time_limit

    # Get prompt or file
    if quiz_format == "prompt":
        session["quiz_prompt"] = request.form.get("prompt_text")
        session["quiz_source_type"] = "prompt"
    elif quiz_format == "upload":
        file = request.files.get("upload_file")
        if file and allowed_file(file.filename):
            try:
                full_text = extract_text_from_uploaded_file(file)
                session["quiz_uploaded_text"] = full_text[:2000]  # Limit to first 10,000 chars
                session["quiz_source_type"] = "upload"
            except Exception as e:
                return render_template("quiz-customization.html", error=str(e))
        else:
            return render_template("quiz-customization.html", error="Invalid or missing file.")
    else:
        return render_template("quiz-customization.html", error="Invalid quiz format.")

    # Reset quiz session counters
    session["score"] = 0
    session["question_count"] = 0
    session["questions"] = []  # Will hold all generated questions
    session["current_question_index"] = 0

    # Generate all questions up front
    questions = []
    for i in range(num_items):
        if quiz_format == "prompt":
            passage = generate_varied_reading_comprehension_text(session["quiz_prompt"], i+1)
        else:
            passage = select_relevant_excerpt_with_variety(session["quiz_uploaded_text"], [q["passage"] for q in questions])
        mcq = generate_multiple_choice_question_with_context(passage, i+1)
        questions.append({
            "passage": passage,
            "question": mcq["question"],
            "choices": mcq["choices"],
            "correct_answer": mcq["correct_answer"]
        })
    session["questions"] = questions

    db = connect_db()
    cursor = db.cursor()

    # Insert quiz metadata
    cursor.execute("""
        INSERT INTO custom_quizzes (creator_username, format, num_items, timed, time_limit, source_type, source_content)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        session["username"],
        quiz_format,
        num_items,
        timed,
        time_limit,
        session.get("quiz_source_type"),
        session.get("quiz_prompt") if quiz_format == "prompt" else session.get("quiz_uploaded_text")
    ))
    quiz_id = cursor.lastrowid

    # Insert each question
    for idx, q in enumerate(questions, start=1):
        cursor.execute("""
            INSERT INTO custom_quiz_questions (quiz_id, question_number, passage, question, choices, correct_answer)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            quiz_id,
            idx,
            q["passage"],
            q["question"],
            json.dumps(q["choices"]),
            q["correct_answer"]
        ))

    db.commit()
    cursor.close()
    db.close()

    return redirect(url_for("quiz"))

@app.route("/quiz", methods=["GET", "POST"])
def quiz():
    if "username" not in session:
        return redirect(url_for("home"))

    # Initialize score and question_count if not present
    if "score" not in session:
        session["score"] = 0
    if "question_count" not in session:
        session["question_count"] = 0
    if "questions" not in session or not session["questions"]:
        return redirect(url_for("quiz_customization"))

    questions = session["questions"]
    idx = session.get("current_question_index", 0)
    num_items = session.get("num_items", 5)

    if request.method == "POST":
        # Save remaining time from form
        if "remaining_time" in request.form:
            try:
                session["remaining_time"] = int(request.form.get("remaining_time", session.get("remaining_time", session.get("time_limit", 0) * 60)))
            except Exception:
                session["remaining_time"] = session.get("time_limit", 0) * 60
        else:
            # On GET, if not set, initialize
            if "remaining_time" not in session or not isinstance(session["remaining_time"], int):
                session["remaining_time"] = int(session.get("time_limit", 0)) * 60

        # Check if user clicked End Exam
        if request.form.get("end_exam") == "1":
            # Mark all remaining questions as wrong
            session["question_count"] = num_items  # All questions attempted
            # Do not increment score for unanswered
            return redirect(url_for("result"))

        selected_answer = request.form.get("answer")
        correct_answer = questions[idx]["correct_answer"]

        if selected_answer == correct_answer:
            session["score"] += 1

        session["question_count"] += 1
        session["current_question_index"] = idx + 1

        if session["question_count"] >= num_items:
            return redirect(url_for("result"))

        idx += 1
    else:
        # On GET, if not set, initialize
        if "remaining_time" not in session:
            session["remaining_time"] = session.get("time_limit", 0) * 60

    # Show current question
    if idx < len(questions):
        q = questions[idx]
        return render_template("quiz.html",
            question={"question_text": q["question"], "options": q["choices"], "correct_answer": q["correct_answer"]},
            username=session["username"],
            score=session["score"],
            generated_text=q["passage"],
            quiz_started=True,
            timed=session.get("timed"),
            time_limit=int(session.get("time_limit", 0)),
            remaining_time=int(session.get("remaining_time", int(session.get("time_limit", 0)) * 60)),
        )
    else:
        return redirect(url_for("result"))

@app.route('/practice_quiz')
def practice_quiz():
    return render_template('practice_quiz.html')

@app.route('/account', methods=['GET', 'POST'])
def account_settings():
    if 'username' not in session:
        return redirect(url_for('home'))

    username = session['username']
    
    db = connect_db()
    cursor = db.cursor(dictionary=True)

    # Fetch current user details including role
    cursor.execute("SELECT user_id, username, email, role FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()

    if request.method == 'POST':
        new_username = request.form['username']
        new_email = request.form['email']
        new_password = request.form['password']

        # Check if new username or email already exists
        cursor.execute("SELECT * FROM users WHERE (username = %s OR email = %s) AND user_id != %s", (new_username, new_email, user['user_id']))
        existing_user = cursor.fetchone()

        if existing_user:
            return render_template('account.html', user=user, message="Username or Email already exists!")

        # Update details in database
        if new_password:
            hashed_password = generate_password_hash(new_password)
            cursor.execute("UPDATE users SET username = %s, email = %s, password = %s WHERE user_id = %s",
                           (new_username, new_email, hashed_password, user['user_id']))
        else:
            cursor.execute("UPDATE users SET username = %s, email = %s WHERE user_id = %s",
                           (new_username, new_email, user['user_id']))

        db.commit()
        session['username'] = new_username  # Update session with new username

        return render_template('account.html', user={'username': new_username, 'email': new_email, 'role': user['role']}, message="Account updated successfully!")

    cursor.close()
    db.close()
    return render_template('account.html', user=user)

@app.route('/create_child_account', methods=['POST'])
def create_child_account():
    if 'username' not in session:
        return redirect(url_for('home'))
    
    username = session['username']
    
    # Check if user has parent role
    db = connect_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT user_id, role FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    
    if not user or user['role'] != 'parent':
        cursor.close()
        db.close()
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        child_username = request.form['child_username']
        child_password = request.form['child_password']
        child_first_name = request.form['child_first_name']
        child_last_name = request.form['child_last_name']
        child_birthdate = request.form['child_birthdate']
        child_year_level = request.form['child_year_level']
        child_school = request.form.get("child_school")
        
        # Check if child username already exists
        cursor.execute("SELECT * FROM users WHERE username = %s", (child_username,))
        existing_child = cursor.fetchone()
        
        if existing_child:
            cursor.close()
            db.close()
            return render_template('account.html', 
                                user={'username': username, 'role': 'parent'}, 
                                message="Child username already taken!")
        
        try:
            # Create kid user account
            hashed_password = generate_password_hash(child_password)
            cursor.execute("""
                INSERT INTO users (username, password, role, first_name, last_name, birthdate)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (child_username, hashed_password, 's.child', child_first_name, child_last_name, child_birthdate))
            
            child_user_id = cursor.lastrowid
            
            # Create child info record
            cursor.execute("""
                INSERT INTO child_info (user_id, first_name, last_name, birthdate, year_level, school)
                VALUES (%s, %s, %s, %s, %s, %s)
                """, (child_user_id, child_first_name, child_last_name, child_birthdate, child_year_level, child_school))
            
            db.commit()
            
            cursor.close()
            db.close()
            
            return render_template('account.html', 
                                user={'username': username, 'role': 'parent'}, 
                                message="Child account created successfully!")
            
        except Exception as e:
            db.rollback()
            cursor.close()
            db.close()
            return render_template('account.html', 
                                user={'username': username, 'role': 'parent'}, 
                                message=f"Error creating child account: {str(e)}")
    
    cursor.close()
    db.close()
    return redirect(url_for('account_settings'))

@app.route("/result")
def result():
    if "username" not in session:
        return redirect(url_for("home"))

    username = session["username"]
    total_questions = session.get("question_count", 0)
    correct_answers = session.get("score", 0)
    current_difficulty = "medium"  # Adjust this dynamically if needed

    # Establish database connection
    db = connect_db()
    cursor = db.cursor()

    # Get user_id based on username
    user_id = get_user_id(username, cursor)
    
    if user_id:
        cursor.execute(
            """
            INSERT INTO user_progress (user_id, total_questions, correct_answers, current_difficulty, last_updated)
            VALUES (%s, %s, %s, %s, NOW())
            """,
            (user_id, total_questions, correct_answers, current_difficulty)
        )
        db.commit()

    cursor.close()
    db.close()

    session.pop("questions", None)
    session.pop("current_question_index", None)
    session.pop("score", None)
    session.pop("question_count", None)
    session.pop("quiz_format", None)
    session.pop("timed", None)
    session.pop("time_limit", None)
    session.pop("quiz_prompt", None)
    session.pop("quiz_uploaded_text", None)
    session.pop("quiz_source_type", None)

    return render_template("result.html", total_questions=total_questions, correct_answers=correct_answers, username = username)

@app.route("/homepage")
def homepage():
    username = session.get("username")
    role = session.get("role")

    db = connect_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT first_name, created_at FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    show_welcome_panel = False
    first_name = None
    if user:
        first_name = user.get("first_name")
        created_at = user.get("created_at")
        if created_at:
            if isinstance(created_at, str):
                from datetime import datetime
                created_at = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
            from datetime import datetime, timedelta
            show_welcome_panel = (datetime.now() - created_at) < timedelta(days=1)

    group_exams = []
    quiz_exams = []
    if not show_welcome_panel:
        # Fetch group exams
        cursor.execute("""
            SELECT g.group_id, g.created_at, s.score
            FROM group_exam_scores s
            JOIN group_exams g ON s.group_id = g.group_id
            WHERE s.username = %s
            ORDER BY g.created_at DESC
        """, (username,))
        group_exams = cursor.fetchall()

        # Fetch individual quizzes from user_progress
        cursor.execute("""
            SELECT total_questions, correct_answers, last_updated
            FROM user_progress
            WHERE user_id = (
                SELECT user_id FROM users WHERE username = %s
            )
            ORDER BY last_updated DESC
        """, (username,))
        quiz_exams = cursor.fetchall()

    cursor.close()
    db.close()
    return render_template(
        "homepage.html",
        first_name=first_name,
        role=role,
        show_welcome_panel=show_welcome_panel,
        group_exams=group_exams,
        quiz_exams=quiz_exams
    )

def get_user_id(username, cursor):
    cursor.execute("SELECT user_id FROM users WHERE username = %s", (username,))
    result = cursor.fetchone()
    return result[0] if result else None

@app.route("/logout")
def logout():
    if "username" in session:
        session.clear()
    return redirect(url_for("home"))

# Function to generate reading comprehension quiz text using DeepSeek via OpenRouter
def generate_reading_comprehension_text(topic):
    messages = [
        {"role": "system", "content": "You write short, simple passages for reading comprehension. Respond with only the passage text."},
        {"role": "user", "content": f"Write a short, concise and simple 1-3 sentence passage about {topic}."},
    ]
    return gemini_complete(messages)

def generate_multiple_choice_question(text):
    messages = [
        {"role": "system", "content": "Create one MCQ from a passage. Respond with ONLY valid JSON: {\"question\": string, \"choices\": [string,string,string,string], \"correct_answer\": string}. No explanations or extra fields."},
        {"role": "user", "content": f"Based on the following text, create the MCQ.\n\n{text}"},
    ]
    generated_text = gemini_complete(messages)
    
    print(f"Raw API response for MCQ: {generated_text[:200]}...")  # Debug logging
    
    try:
        # First try to parse directly as JSON
        mcq = json.loads(generated_text)
    except json.JSONDecodeError:
        try:
            # If direct parsing fails, try to extract JSON from the response
            # Look for JSON patterns in the text
            import re
            # Try to find JSON object in the response
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                mcq = json.loads(json_str)
                print(f"Extracted JSON from response: {json_str[:100]}...")
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            print(f"Failed to parse JSON from Gemini response: {e}")
            print(f"Full response was: {generated_text}")
            # Create a fallback question based on the text
            mcq = {
                "question": f"What is the main topic of this text?",
                "choices": [
                    "A general topic",
                    "The specific subject mentioned",
                    "Something unrelated", 
                    "Not enough information"
                ],
                "correct_answer": "The specific subject mentioned"
            }
    
    # Validate the MCQ structure
    if not mcq.get("question") or not mcq.get("choices") or not mcq.get("correct_answer"):
        print(f"Invalid MCQ structure: {mcq}")
        mcq = {
            "question": f"Based on the text: {text[:50]}...",
            "choices": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": "Option A"
        }
    
    return mcq

@app.route('/generate_quiz_text', methods=['POST'])
def generate_quiz_text():
    data = request.json
    topic = data.get('topic')
    
    if not topic:
        return jsonify({"error": "Topic is required"}), 400
    
    try:
        generated_text = generate_reading_comprehension_text(topic)
        mcq = generate_multiple_choice_question(generated_text)
        return jsonify({
            "generated_text": generated_text,
            "topic": topic,
            "question": mcq.get("question"),
            "choices": mcq.get("choices"),
            "correct_answer": mcq.get("correct_answer")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def select_relevant_excerpt(full_text):
    """
    Use AI to select a short, relevant excerpt from the document text.
    """
    messages = [
        {"role": "system", "content": "From the provided text, return only a short self-contained excerpt (1-4 sentences). No explanations."},
        {"role": "user", "content": f"Select an excerpt from this text:\n\n{full_text}"},
    ]
    return gemini_complete(messages)

@app.route('/upload_document', methods=['POST'])
def upload_document():
    if 'document' not in request.files:
        return jsonify({"error": "No document uploaded"}), 400

    file = request.files['document']

    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Allowed: pdf, docx, txt"}), 400

    try:
        # Extract full text
        full_text = extract_text_from_uploaded_file(file)

        # Store the full document text in session for future question generation
        session["full_document_text"] = full_text

        # Let AI select an excerpt
        excerpt = select_relevant_excerpt(full_text)

        # Generate MCQ from excerpt
        mcq = generate_multiple_choice_question(excerpt)

        return jsonify({
            "generated_text": excerpt,
            "question": mcq.get("question"),
            "choices": mcq.get("choices"),
            "correct_answer": mcq.get("correct_answer")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/admin")
@admin_required
def admin_dashboard():
    return render_template("admin.html")

@app.route("/group_exam")
def group_exam():
    if 'username' not in session:
        return redirect(url_for('home'))
    return render_template("group_exam.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    success = None

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        user_type = request.form.get("user_type")
        first_name = request.form.get("first_name")
        middle_name = request.form.get("middle_name")
        last_name = request.form.get("last_name")
        email = request.form.get("email")
        birthdate = request.form.get("birthdate")
        contact_no = request.form.get("contact_no")


        if password != confirm_password:
            error = "Passwords do not match."
        else:
            conn = connect_db()
            cursor = conn.cursor(dictionary=True)

            # Check if username already exists
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            existing_user = cursor.fetchone()

            if existing_user:
                error = "Username already taken."
            else:
                hashed_password = generate_password_hash(password)
                cursor.execute("""
                    INSERT INTO users (username, password, role, first_name, middle_name, last_name, email, birthdate, contact_no)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (username, hashed_password, user_type, first_name, middle_name, last_name, email, birthdate, contact_no))
                user_id = cursor.lastrowid


                conn.commit()
                success = "Registration successful! You can now log in."

            conn.close()

    return render_template("register.html", error=error, success=success)

@app.route('/get_users', methods=['GET'])
def get_users():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, username, password, role, email, birthdate, created_at FROM users")
    users = cursor.fetchall()
    conn.close()
    user_list = [
        {
            "user_id": row[0],
            "username": row[1],
            "password": row[2],
            "role": row[3],
            "email":row[4],
            "birthdate": row[5],
            "created_at": row[6]
        }
        for row in users
    ]

    return jsonify(user_list)

@app.route('/add_user', methods=['POST'])
def add_user():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    role = data.get("role")
    email = data.get("email")
    birthdate = data.get("birthdate")

    if not username or not password or not role or not email or not birthdate:
        return jsonify({"error": "All fields are required"}), 400

    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO users (username, password, role, email, birthdate) 
        VALUES (%s, %s, %s, %s, %s)
    """, (username, password, role, email, birthdate))
    conn.commit()
    conn.close()

    return jsonify({"message": "User added successfully"}), 201

@app.route('/edit_user/<int:user_id>', methods=['PUT'])
def edit_user(user_id):
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400

        username = data.get("username")
        role = data.get("role")
        email = data.get("email")
        birthdate = data.get("birthdate")

        if not username or not role or not email or not birthdate:
            return jsonify({"error": "All fields are required"}), 400

        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE users 
            SET username = %s, role = %s, email = %s, birthdate = %s 
            WHERE user_id = %s
        """, (username, role, email, birthdate, user_id))
        conn.commit()
        conn.close()

        return jsonify({"message": "User updated successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/delete_user/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
    conn.commit()
    conn.close()

    return jsonify({"message": "User deleted successfully"})

@app.route('/get_questions', methods=['GET'])
def get_questions():
    """
    Retrieve all questions from custom_quiz_questions and group_exam_questions,
    including the creator (username) for each question.
    Returns a JSON list with a 'source' field and 'creator' field.
    """
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    # Fetch from custom_quiz_questions and join with custom_quizzes to get creator
    cursor.execute("""
        SELECT 
            q.question_id, q.quiz_id, q.question_number, q.passage, q.question, q.choices, q.correct_answer, 
            'custom_quiz' AS source,
            cq.creator_username AS creator
        FROM custom_quiz_questions q
        JOIN custom_quizzes cq ON q.quiz_id = cq.quiz_id
    """)
    custom_questions = cursor.fetchall()

    # Fetch from group_exam_questions and join with group_exams to get host (creator)
    cursor.execute("""
        SELECT 
            q.id AS question_id, q.group_id, q.question_number, q.passage, q.question, q.choices, q.correct_answer, 
            'group_exam' AS source,
            ge.host_username AS creator
        FROM group_exam_questions q
        JOIN group_exams ge ON q.group_id = ge.group_id
    """)
    group_questions = cursor.fetchall()

    cursor.close()
    conn.close()

    # Combine and return as JSON
    all_questions = custom_questions + group_questions
    return jsonify(all_questions)

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('home'))

    username = session['username']
    
    # Establish database connection
    db = connect_db()
    cursor = db.cursor()

    # Get user_id based on username
    user_id = get_user_id(username, cursor)

    if user_id:
        # Count total quizzes completed
        cursor.execute("SELECT COUNT(*) FROM user_progress WHERE user_id = %s", (user_id,))
        quizzes_completed = cursor.fetchone()[0] or 0

        # Calculate average score
        cursor.execute("SELECT AVG(correct_answers / total_questions * 100) FROM user_progress WHERE user_id = %s AND total_questions > 0", (user_id,))
        avg_score = cursor.fetchone()[0]
        avg_score = round(avg_score, 2) if avg_score is not None else 0

        # Get current difficulty (most recent quiz difficulty)
        cursor.execute("""
            SELECT current_difficulty 
            FROM user_progress 
            WHERE user_id = %s 
            ORDER BY last_updated DESC LIMIT 1
        """, (user_id,))
        current_difficulty = cursor.fetchone()
        current_difficulty = current_difficulty[0] if current_difficulty else "N/A"  # Default to "N/A" if no quizzes taken
    else:
        quizzes_completed = 0
        avg_score = 0
        current_difficulty = "N/A"

    cursor.close()
    db.close()

    return render_template('dashboard.html', username=username, 
                           quizzes_completed=quizzes_completed, 
                           average_score=avg_score,
                           current_difficulty=current_difficulty)
#Plan: Integrate dashboard features to homepage or somewhere else, obsolete page


@app.route('/set_generated_question', methods=['POST'])
def set_generated_question():
    data = request.json
    question_text = data.get("question")
    choices = data.get("choices")
    correct_answer = data.get("correct_answer")
    generated_text = data.get("generated_text")
    use_custom_source = data.get("use_custom_source", False)
    topic = data.get("topic")

    if not question_text or not choices or not correct_answer:
        return jsonify({"error": "Missing question data"}), 400

    session["generated_question"] = question_text
    session["generated_choices"] = choices
    session["correct_answer"] = correct_answer
    
    if use_custom_source and generated_text:
        session["custom_quiz_active"] = True
        session["quiz_source_type"] = "uploaded_file"
        # Store both the excerpt (for display) and the full text (for future excerpts)
        if "full_document_text" in session:
            session["generated_text"] = generated_text  # current excerpt shown
        else:
            # fallback, keep at least the excerpt
            session["full_document_text"] = generated_text
            session["generated_text"] = generated_text

    elif topic and not use_custom_source:
        # For prompt-based quizzes: store the topic and mark as topic-based
        session["current_topic"] = topic
        session["topic_quiz_active"] = True
        session["quiz_source_type"] = "prompt_topic"
        # Clear uploaded file quiz state
        session.pop("custom_quiz_active", None)
        session.pop("source_text", None)
    
    # Always store the current generated text for display
    if generated_text:
        session["generated_text"] = generated_text

    return jsonify({"message": "Generated question set in session"})

def select_relevant_excerpt_with_variety(full_text, used_excerpts=None):
    """
    Use AI to select a short, relevant excerpt from the document text,
    ensuring variety by avoiding previously used excerpts.
    """
    if used_excerpts is None:
        used_excerpts = []
    
    # Create a prompt that encourages variety
    variety_instruction = ""
    if used_excerpts:
        variety_instruction = f"\n\nPreviously used excerpts to AVOID repeating:\n" + "\n".join([f"- {excerpt[:100]}..." for excerpt in used_excerpts[-3:]])  # Show last 3 to avoid
    
    messages = [
        {
            "role": "system", 
            "content": f"From the provided text, return only a short self-contained excerpt (2-4 sentences) that is DIFFERENT from any previously used excerpts. Focus on a different aspect, topic, or section of the document. No explanations.{variety_instruction}"
        },
        {"role": "user", "content": f"Select a NEW and DIFFERENT excerpt from this text:\n\n{full_text}"},
    ]
    return gemini_complete(messages)

def generate_varied_reading_comprehension_text(topic, question_number=1, previous_passages=None):
    """
    Generate reading comprehension text with variety based on question number and previous content.
    """
    if previous_passages is None:
        previous_passages = []
    
    # Create different aspects/angles for the same topic
    aspects = [
        f"basic introduction to {topic}",
        f"interesting facts about {topic}",
        f"how {topic} works or functions",
        f"the history or origin of {topic}",
        f"the importance or benefits of {topic}",
        f"different types or categories of {topic}",
        f"recent discoveries or developments about {topic}",
        f"practical applications of {topic}"
    ]
    
    # Select a different aspect based on question number
    aspect_index = (question_number - 1) % len(aspects)
    chosen_aspect = aspects[aspect_index]
    
    # Add instruction to avoid repetition if we have previous passages
    variety_instruction = ""
    if previous_passages:
        variety_instruction = f"\n\nPreviously covered content to AVOID repeating:\n" + "\n".join([f"- {passage[:80]}..." for passage in previous_passages[-2:]])
    
    messages = [
        {
            "role": "system", 
            "content": f"You write short, simple passages for reading comprehension. Focus on {chosen_aspect}. Make each passage unique and different from previous ones. Respond with only the passage text (2-4 sentences).{variety_instruction}"
        },
        {"role": "user", "content": f"Write a short, concise passage focusing on {chosen_aspect}."},
    ]
    return gemini_complete(messages)

def generate_multiple_choice_question_with_context(text, question_number=1):
    """
    Generate MCQ with more variety based on question number and text content.
    """
    # Different question types to encourage variety
    question_types = [
        "main idea or topic",
        "specific detail mentioned",
        "cause and effect relationship",
        "comparison or contrast",
        "sequence or process",
        "purpose or function",
        "characteristic or feature",
        "location or setting"
    ]
    
    question_type = question_types[(question_number - 1) % len(question_types)]
    
    messages = [
        {
            "role": "system", 
            "content": f"Create one MCQ focusing on {question_type} from a passage. Respond with ONLY valid JSON: {{\"question\": string, \"choices\": [string,string,string,string], \"correct_answer\": string}}. Make the question and all choices clearly distinct and avoid generic options."
        },
        {"role": "user", "content": f"Based on the following text, create an MCQ about {question_type}:\n\n{text}"},
    ]
    
    generated_text = gemini_complete(messages)
    
    print(f"Raw API response for MCQ (Question #{question_number}): {generated_text[:200]}...")
    
    try:
        mcq = json.loads(generated_text)
    except json.JSONDecodeError:
        try:
            import re
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                mcq = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            print(f"Failed to parse JSON from Gemini response: {e}")
            # Create a more varied fallback based on question number
            fallback_questions = [
                f"What is the main subject discussed in this text?",
                f"Which detail is mentioned in the passage?",
                f"What relationship is described in the text?",
                f"According to the passage, what happens?",
                f"What characteristic is highlighted in the text?"
            ]
            fallback_question = fallback_questions[(question_number - 1) % len(fallback_questions)]
            
            mcq = {
                "question": fallback_question,
                "choices": [
                    "First possible answer",
                    "Second possible answer", 
                    "Third possible answer",
                    "Fourth possible answer"
                ],
                "correct_answer": "First possible answer"
            }
    
    # Validate the MCQ structure
    if not mcq.get("question") or not mcq.get("choices") or not mcq.get("correct_answer"):
        print(f"Invalid MCQ structure: {mcq}")
        mcq = {
            "question": f"Question #{question_number}: What does this passage discuss?",
            "choices": ["Topic A", "Topic B", "Topic C", "Topic D"],
            "correct_answer": "Topic A"
        }
    return mcq

@app.route('/submit_answer_and_generate_new', methods=['POST'])
def submit_answer_and_generate_new():
    """Submit answer and generate new content with improved variety"""
    if "username" not in session:
        return jsonify({"error": "Not logged in"}), 401

    data = request.json
    selected_answer = data.get("answer")

    if not selected_answer:
        return jsonify({"error": "No answer provided"}), 400

    # Check if answer is correct
    correct_answer = session.get("correct_answer")
    is_correct = selected_answer == correct_answer

    # Update score
    if is_correct:
        session["score"] = session.get("score", 0) + 1

    # Increment question count
    session["question_count"] = session.get("question_count", 0) + 1
    current_question_number = session["question_count"] + 1  # For next question

    # Check if quiz is finished
    if session["question_count"] >= 5:
        return jsonify({
            "score": session["score"],
            "quiz_finished": True
        })

    # Store the current passage in history for variety checking
    current_passage = session.get("generated_text", "")
    if current_passage:
        # Initialize or update passage history
        if "passage_history" not in session:
            session["passage_history"] = []
        session["passage_history"].append(current_passage)
        # Keep only last 3 passages to avoid memory buildup
        session["passage_history"] = session["passage_history"][-3:]

    # Clear current question data before generating new content
    session.pop("generated_question", None)
    session.pop("generated_choices", None)
    session.pop("correct_answer", None)

    # Generate new content with variety
    try:
        quiz_source_type = session.get("quiz_source_type")
        
        if quiz_source_type == "uploaded_file" and session.get("full_document_text"):
            # Pick a NEW excerpt from the full document each time
            full_text = session.get("full_document_text")
            used_excerpts = session.get("passage_history", [])
            
            # Try multiple times to get a different excerpt
            max_attempts = 3
            for attempt in range(max_attempts):
                excerpt = select_relevant_excerpt_with_variety(full_text, used_excerpts)
                
                # Check if this excerpt is significantly different from previous ones
                is_different = True
                for used_excerpt in used_excerpts[-2:]:  # Check last 2
                    if len(excerpt) > 0 and len(used_excerpt) > 0:
                        # Simple similarity check - if more than 70% of words are the same, it's too similar
                        excerpt_words = set(excerpt.lower().split())
                        used_words = set(used_excerpt.lower().split())
                        if len(excerpt_words & used_words) / max(len(excerpt_words), 1) > 0.7:
                            is_different = False
                            break
                
                if is_different or attempt == max_attempts - 1:
                    break
            
            mcq = generate_multiple_choice_question_with_context(excerpt, current_question_number)

            # Update session with new content
            session["generated_text"] = excerpt
            session["generated_question"] = mcq.get("question")
            session["generated_choices"] = mcq.get("choices")
            session["correct_answer"] = mcq.get("correct_answer")

            new_content = {
                "generated_text": excerpt,
                "question": mcq.get("question"),
                "choices": mcq.get("choices"),
                "correct_answer": mcq.get("correct_answer"),
                "topic": "Document Content"
            }

            return jsonify({
                "score": session["score"],
                "quiz_finished": False,
                "new_content": new_content,
                "use_custom_source": True
            })

        elif quiz_source_type == "prompt_topic" and session.get("current_topic"):
            # For topic-based quizzes, generate varied content
            current_topic = session.get("current_topic")
            previous_passages = session.get("passage_history", [])
            
            generated_text = generate_varied_reading_comprehension_text(
                current_topic, current_question_number, previous_passages
            )
            mcq = generate_multiple_choice_question_with_context(generated_text, current_question_number)
            
            # Update session with new content
            session["generated_text"] = generated_text
            session["generated_question"] = mcq.get("question")
            session["generated_choices"] = mcq.get("choices")
            session["correct_answer"] = mcq.get("correct_answer")
            
            new_content = {
                "generated_text": generated_text,
                "question": mcq.get("question"),
                "choices": mcq.get("choices"),
                "correct_answer": mcq.get("correct_answer"),
                "topic": current_topic
            }
            
            return jsonify({
                "score": session["score"],
                "quiz_finished": False,
                "new_content": new_content,
                "use_custom_source": False
            })
            
        else:
            # Fallback with variety
            fallback_topic = "general knowledge"
            previous_passages = session.get("passage_history", [])
            
            generated_text = generate_varied_reading_comprehension_text(
                fallback_topic, current_question_number, previous_passages
            )
            mcq = generate_multiple_choice_question_with_context(generated_text, current_question_number)
            
            # Update session with new content
            session["generated_text"] = generated_text
            session["generated_question"] = mcq.get("question")
            session["generated_choices"] = mcq.get("choices")
            session["correct_answer"] = mcq.get("correct_answer")
            session["current_topic"] = fallback_topic
            session["quiz_source_type"] = "prompt_topic"
            
            new_content = {
                "generated_text": generated_text,
                "question": mcq.get("question"),
                "choices": mcq.get("choices"),
                "correct_answer": mcq.get("correct_answer"),
                "topic": fallback_topic
            }
            
            return jsonify({
                "score": session["score"],
                "quiz_finished": False,
                "new_content": new_content,
                "use_custom_source": False
            })
            
    except Exception as e:
        print(f"Error generating new content: {str(e)}")
        return jsonify({"error": f"Failed to generate new content: {str(e)}"}), 500

@app.route('/start_quiz')
def start_quiz():
    if "username" not in session:
        return redirect(url_for("home"))
    # Reset quiz session counters
    session["score"] = 0
    session["question_count"] = 0
    # Clear any existing question data
    session.pop("generated_question", None)
    session.pop("generated_choices", None)
    session.pop("correct_answer", None)
    session.pop("generated_text", None)
    session.pop("custom_quiz_active", None)
    session.pop("source_text", None)
    session.pop("current_topic", None)
    session.pop("quiz_source_type", None)
    session.pop("topic_quiz_active", None)
    session.pop("passage_history", None)  # Clear passage history for variety
    session.pop("full_document_text", None)  # Clear document text
    return redirect(url_for("quiz"))

@app.route("/start_group_exam", methods=["POST"])
def start_group_exam():
    group_id = request.form.get("group_id")
    quiz_format = request.form.get("format")
    num_items = int(request.form.get("num_items", 5))
    timed = request.form.get("timed")
    time_limit = request.form.get("time_limit") if timed == "yes" else None
    host = session.get("username")

    # Get prompt or file
    if quiz_format == "prompt":
        prompt_text = request.form.get("prompt_text")
        source_type = "prompt"
        source_content = prompt_text
    elif quiz_format == "upload":
        file = request.files.get("upload_file")
        if file and allowed_file(file.filename):
            source_type = "upload"
            source_content = extract_text_from_uploaded_file(file)
        else:
            return render_template("group_exam.html", error="Invalid or missing file.")
    else:
        return render_template("group_exam.html", error="Invalid quiz format.")

    # Generate questions
    questions = []
    for i in range(num_items):
        if quiz_format == "prompt":
            passage = generate_varied_reading_comprehension_text(source_content, i+1)
        else:
            passage = select_relevant_excerpt_with_variety(source_content, [q["passage"] for q in questions])
        mcq = generate_multiple_choice_question_with_context(passage, i+1)
        questions.append({
            "passage": passage,
            "question": mcq["question"],
            "choices": json.dumps(mcq["choices"]),  # Store as JSON string
            "correct_answer": mcq["correct_answer"]
        })

    # Save group exam to DB
    db = connect_db()
    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO group_exams (group_id, host_username, format, num_items, timed, time_limit, source_type, source_content)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (group_id, host, quiz_format, num_items, timed, time_limit, source_type, source_content))
    db.commit()

    # Save questions to DB
    for idx, q in enumerate(questions, start=1):
        cursor.execute("""
            INSERT INTO group_exam_questions (group_id, question_number, passage, question, choices, correct_answer)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (group_id, idx, q["passage"], q["question"], q["choices"], q["correct_answer"]))
    db.commit()

    db.commit()
    cursor.close()
    db.close()

    session["group_id"] = group_id
    return redirect(url_for("group_room"))

@app.route('/join_group_exam', methods=['POST'])
def join_group_exam():
    group_id = request.form.get('group_id')
    username = session.get('username')

    db = connect_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM group_exams WHERE group_id = %s", (group_id,))
    group = cursor.fetchone()

    if group:
        # Add user to group_exam_users if not already present
        cursor.execute("SELECT * FROM group_exam_users WHERE group_id = %s AND username = %s", (group_id, username))
        user_in_group = cursor.fetchone()
        if not user_in_group:
            cursor.execute("""
                INSERT INTO group_exam_users (group_id, username, finished)
                VALUES (%s, %s, %s)
            """, (group_id, username, False))
            db.commit()
        session['group_id'] = group_id
        cursor.close()
        db.close()
        return redirect(url_for('group_exam_take'))
    else:
        cursor.close()
        db.close()
        return render_template('homepage.html', error="Group ID not found.", first_name=session.get('first_name'), role=session.get('role'))

@app.route('/group_exam_take', methods=['GET', 'POST'])
def group_exam_take():
    if "username" not in session or "group_id" not in session:
        return redirect(url_for("home"))

    group_id = session["group_id"]
    username = session["username"]

    db = connect_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM group_exams WHERE group_id = %s", (group_id,))
    group = cursor.fetchone()
    if not group:
        cursor.close()
        db.close()
        return redirect(url_for("homepage"))

    num_items = group["num_items"]
    cursor.execute("SELECT * FROM group_exam_questions WHERE group_id = %s ORDER BY question_number ASC", (group_id,))
    questions = cursor.fetchall()

    # Track progress in session
    if "group_exam_score" not in session:
        session["group_exam_score"] = 0
    if "group_exam_question_count" not in session:
        session["group_exam_question_count"] = 0
    if "group_exam_current_index" not in session:
        session["group_exam_current_index"] = 0

    idx = session.get("group_exam_current_index", 0)

    if request.method == "POST":
        selected_answer = request.form.get("answer")
        correct_answer = questions[idx]["correct_answer"]

        if selected_answer == correct_answer:
            session["group_exam_score"] += 1

        session["group_exam_question_count"] += 1
        session["group_exam_current_index"] = idx + 1

        if session["group_exam_question_count"] >= num_items:
            # Save score to DB
            cursor.execute("""
                INSERT INTO group_exam_scores (group_id, username, score)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE score = %s, finished_at = NOW()
            """, (group_id, username, session["group_exam_score"], session["group_exam_score"]))
            # Mark user as finished
            cursor.execute("""
                UPDATE group_exam_users SET finished = TRUE WHERE group_id = %s AND username = %s
            """, (group_id, username))
            db.commit()

            # Fetch score from DB for display
            cursor.execute("SELECT score FROM group_exam_scores WHERE group_id = %s AND username = %s", (group_id, username))
            score_row = cursor.fetchone()
            score = score_row["score"] if score_row else session.get("group_exam_score", 0)

            # Clear user progress for next group exam
            session.pop("group_exam_score", None)
            session.pop("group_exam_question_count", None)
            session.pop("group_exam_current_index", None)

            cursor.close()
            db.close()
            return render_template(
                "result.html",
                total_questions=num_items,
                correct_answers=score,
                username=username
            )

        idx += 1
    else:
        # On GET, if not set, initialize
        if "remaining_time" not in session:
            session["remaining_time"] = session.get("time_limit", 0) * 60

    # Show current question
    if idx < len(questions):
        q = questions[idx]
        choices = json.loads(q["choices"])
        cursor.close()
        db.close()
        return render_template(
            "group_exam_take.html",
            question={"question_text": q["question"], "options": choices, "correct_answer": q["correct_answer"]},
            username=username,
            score=session.get("group_exam_score", 0),
            generated_text=q["passage"],
            group_id=group_id,
            time_limit=int(group.get("time_limit", 0)),  # <-- FIX: get from group, not session
            remaining_time=int(session.get("remaining_time", int(group.get("time_limit", 0)) * 60)),
            timed=group.get("timed", "no"),              # <-- FIX: get from group, not session
        )
    else:
        # If all questions answered, show result
        cursor.close()
        db.close()
        return render_template(
            "result.html",
            total_questions=num_items,
            correct_answers=session.get("group_exam_score", 0),
            username=username
        )

@app.route("/group_room")
def group_room():
    if "username" not in session:
        return redirect(url_for("home"))
    group_id = session.get("group_id")

    db = connect_db()
    cursor = db.cursor(dictionary=True)

    # Get group info
    cursor.execute("SELECT * FROM group_exams WHERE group_id = %s", (group_id,))
    group = cursor.fetchone()
    if not group:
        cursor.close()
        db.close()
        return redirect(url_for("homepage"))

    # Get users (excluding host)
    cursor.execute("SELECT username FROM group_exam_users WHERE group_id = %s AND username != %s", (group_id, group["host_username"]))
    users = [row["username"] for row in cursor.fetchall()]

    # Get questions
    cursor.execute("SELECT * FROM group_exam_questions WHERE group_id = %s ORDER BY question_number ASC", (group_id,))
    questions = cursor.fetchall()
    for q in questions:
        q["choices"] = json.loads(q["choices"])

    # Get scores
    cursor.execute("SELECT username, score FROM group_exam_scores WHERE group_id = %s", (group_id,))
    scores = {row["username"]: row["score"] for row in cursor.fetchall()}

    cursor.close()
    db.close()

    return render_template(
        "group_room.html",
        group_id=group_id,
        users=users,
        questions=questions,
        scores=scores,
        host=group["host_username"],
        username=session["username"],
        time_limit=int(group.get("time_limit", 0)),   # <-- Add this
        timed=group.get("timed", "no"),               # <-- Add this
        remaining_time=int(session.get("remaining_time", int(group.get("time_limit", 0)) * 60)),  # Optional for timer
    )

app.config["GOOGLE_OAUTH_CLIENT_ID"] = os.environ.get("GOOGLE_OAUTH_CLIENT_ID")
app.config["GOOGLE_OAUTH_CLIENT_SECRET"] = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET")

# Remove the duplicate app config lines and fix the blueprint
google_bp = make_google_blueprint(
    client_id=os.environ.get("GOOGLE_OAUTH_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET"),
    scope=["openid", "https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile"],
    redirect_to="oauth_success"  # Changed this - redirect to homepage after successful auth
)
app.register_blueprint(google_bp, url_prefix="/login")

# Remove the custom authorized route and use Flask-Dance's built-in handling
# Instead, create a handler that checks for successful OAuth
@app.route('/oauth_success')
def oauth_success():
    """Handle successful OAuth login"""
    if not google.authorized:
        return redirect(url_for("home"))
    
    try:
        # Get user info from Google
        resp = google.get("/oauth2/v2/userinfo")
        if not resp.ok:
            print(f"Failed to fetch user info: {resp.status_code}")
            return redirect(url_for("home"))

        user_info = resp.json()
        print(f"User info received: {user_info}")
        
        email = user_info.get("email")
        name = user_info.get("name", "")
        given_name = user_info.get("given_name", "")
        
        if not email:
            print("No email found in user info")
            return redirect(url_for("home"))

        # Database operations
        db = connect_db()
        cursor = db.cursor(dictionary=True)
        
        try:
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            user = cursor.fetchone()

            if not user:
                # Create new user
                username = email.split("@")[0]
                cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
                if cursor.fetchone():
                    username = f"{username}_{random.randint(1000, 9999)}"
                
                cursor.execute("""
                    INSERT INTO users (username, email, role, first_name, password)
                    VALUES (%s, %s, %s, %s, %s)
                """, (username, email, "student", given_name or name, "oauth_google"))
                
                db.commit()
                session_username = username
                role = "student"
            else:
                session_username = user["username"]
                role = user["role"]

            # Set session
            session["username"] = session_username
            session["role"] = role
            session["guest"] = False
            session["score"] = 0
            session["question_count"] = 0
            
            print(f"User logged in: {session_username}")
            
        except Exception as db_error:
            print(f"Database error: {db_error}")
            db.rollback()
            return redirect(url_for("home"))
        finally:
            cursor.close()
            db.close()
            
        return redirect(url_for("homepage"))
        
    except Exception as e:
        print(f"OAuth error: {e}")
        return redirect(url_for("home"))

# Add a simple Google login route
@app.route('/login_google')
def login_google():
    """Initiate Google login or handle if already authorized"""
    if google.authorized:
        return redirect(url_for("oauth_success"))
    return redirect(url_for("google.login"))

@app.route('/end_group_exam', methods=['POST'])
def end_group_exam():
    group_id = request.form.get('group_id')
    db = connect_db()
    cursor = db.cursor(dictionary=True)

    # Mark exam as ended
    cursor.execute("UPDATE group_exams SET exam_ended = %s WHERE group_id = %s", (True, group_id))

    # Get exam info
    cursor.execute("SELECT num_items FROM group_exams WHERE group_id = %s", (group_id,))
    group = cursor.fetchone()
    num_items = group["num_items"] if group else 0

    # Get all users in the group
    cursor.execute("SELECT username FROM group_exam_users WHERE group_id = %s", (group_id,))
    users = [row["username"] for row in cursor.fetchall()]

    for username in users:
        # Get current score and questions answered for this user
        cursor.execute("SELECT score FROM group_exam_scores WHERE group_id = %s AND username = %s", (group_id, username))
        score_row = cursor.fetchone()
        current_score = score_row["score"] if score_row else 0

        cursor.execute("SELECT finished FROM group_exam_users WHERE group_id = %s AND username = %s", (group_id, username))
        finished_row = cursor.fetchone()
        finished = finished_row["finished"] if finished_row else False

        if not finished:
            # Count how many questions answered (if you track this, otherwise assume all unanswered are wrong)
            cursor.execute("SELECT COUNT(*) AS answered FROM group_exam_scores WHERE group_id = %s AND username = %s", (group_id, username))
            _ = cursor.fetchone()  # <-- Always fetch the result, even if unused

            # Save score (unanswered = wrong, so score stays as is)
            cursor.execute("""
                INSERT INTO group_exam_scores (group_id, username, score)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE score = %s, finished_at = NOW()
            """, (group_id, username, current_score, current_score))

            # Mark user as finished
            cursor.execute("""
                UPDATE group_exam_users SET finished = TRUE WHERE group_id = %s AND username = %s
            """, (group_id, username))

    db.commit()
    cursor.close()
    db.close()
    return redirect(url_for('homepage'))

@app.route('/add_group_exam_time', methods=['POST'])
def add_group_exam_time():
    data = request.get_json()
    group_id = data.get('group_id')
    minutes = int(data.get('minutes', 0))
    db = connect_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT time_limit FROM group_exams WHERE group_id = %s", (group_id,))
    group = cursor.fetchone()
    if not group:
        cursor.close()
        db.close()
        return jsonify(success=False, error="Group not found"), 404
    # Add minutes to current time_limit
    new_time_limit = int(group["time_limit"] or 0) + minutes
    cursor.execute("UPDATE group_exams SET time_limit = %s WHERE group_id = %s", (new_time_limit, group_id))
    db.commit()
    cursor.close()
    db.close()
    return jsonify(success=True, remaining_time=new_time_limit * 60)
    

@app.route('/records')
def records():
    if session.get('role') != 'teacher':
        return redirect(url_for('homepage'))

    db = connect_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM group_exams ORDER BY created_at DESC")
    exams = cursor.fetchall()
    group_exams = []
    for exam in exams:
        cursor.execute("SELECT username, score FROM group_exam_scores WHERE group_id = %s", (exam['group_id'],))
        examinees = cursor.fetchall()
        group_exams.append({
            "group_id": exam['group_id'],
            "host_username": exam['host_username'],
            "created_at": exam.get('created_at'),
            "source_content": exam.get('source_content', ''),  # <-- Make sure this is included
            "examinees": examinees
        })
    cursor.close()
    db.close()
    return render_template('records.html', group_exams=group_exams)

@app.route('/check_exam_ended')
def check_exam_ended():
    group_id = session.get("group_id")
    if not group_id:
        return jsonify({"error": "No group_id"}), 400
    db = connect_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT exam_ended FROM group_exams WHERE group_id = %s", (group_id,))
    row = cursor.fetchone()
    cursor.close()
    db.close()
    if row:
        return jsonify({"exam_ended": bool(row["exam_ended"])})
    return jsonify({"exam_ended": False})

@app.route('/debug/session')
def debug_session():
    return {
        'session': dict(session),
        'google_authorized': google.authorized if 'google' in globals() else 'Not available'
    }

if __name__ == "__main__":
    app.run(debug=False,  ssl_context = 'adhoc')
