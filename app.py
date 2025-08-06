from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from werkzeug.utils import secure_filename
from datetime import datetime
from reportlab.pdfgen import canvas
from io import BytesIO
import os
import sqlite3
import numpy as np
import uuid
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
app.secret_key = 'an@s_secure_key_123456789!'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# ---------------- MODEL SETUP ----------------
MODEL_URL = "https://huggingface.co/cozwatt/skin-disease-detector/resolve/main/skin_model.h5"
MODEL_DIR = "train_model"
MODEL_PATH = os.path.join(MODEL_DIR, "skin_model.h5")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.txt")

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Download model if not already present
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Hugging Face...")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("✅ Model downloaded!")

download_model()

# Load the model
model = load_model(MODEL_PATH)

# Load class names
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = [line.strip() for line in f]

# ---------------- DATABASE INIT ----------------
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        image_path TEXT,
        result TEXT,
        confidence REAL,
        date TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ---------------- ROUTES ----------------
@app.route('/')
def home():
    if 'user_id' in session:
        return render_template('index.html', username=session.get('username'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('database.db')
        try:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            flash("Registered successfully! Please login.", "success")
            return redirect(url_for('login'))
        except:
            flash("Username already exists.", "danger")
        finally:
            conn.close()
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()
        conn.close()
        if user:
            session['user_id'] = user[0]
            session['username'] = username
            return redirect(url_for('home'))
        else:
            flash("Invalid credentials", "danger")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if 'image' not in request.files:
        flash("No image uploaded", "warning")
        return redirect(url_for('home'))

    image = request.files['image']
    if image.filename == '':
        flash("No selected file", "warning")
        return redirect(url_for('home'))

    filename = secure_filename(str(uuid.uuid4()) + "_" + image.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)

    img = load_img(filepath, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    confidence = round(100 * np.max(predictions), 2)
    result = class_names[np.argmax(predictions)]

    # Save to DB
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("INSERT INTO predictions (user_id, image_path, result, confidence, date) VALUES (?, ?, ?, ?, ?)",
              (session['user_id'], filename, result, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    prediction_id = c.lastrowid
    conn.commit()
    conn.close()

    return render_template('result.html',
        result=result,
        confidence=confidence,
        image_filename=filename,
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        prediction_id=prediction_id
    )

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT id, image_path, result, confidence, date FROM predictions WHERE user_id=? ORDER BY id DESC", (session['user_id'],))
    records = [
        {"id": row[0], "image_path": row[1], "result": row[2], "confidence": row[3], "date": row[4]}
        for row in c.fetchall()
    ]
    conn.close()
    return render_template('history.html', history=records)

@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT result, COUNT(*) FROM predictions GROUP BY result")
    stats = c.fetchall()
    conn.close()
    return render_template('dashboard.html', stats=stats)

@app.route('/download_pdf/<int:id>')
def download_pdf(id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT image_path, result, confidence, date FROM predictions WHERE id=? AND user_id=?", (id, session['user_id']))
    record = c.fetchone()
    conn.close()

    if not record:
        flash("Prediction not found.", "warning")
        return redirect(url_for('history'))

    image_path, result, confidence, date = record

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.setTitle("Skin Disease Prediction Report")

    pdf.drawString(50, 800, "Skin Disease Prediction Report")
    pdf.drawString(50, 770, f"User: {session['username']}")
    pdf.drawString(50, 750, f"Date: {date}")
    pdf.drawString(50, 730, f"Prediction: {result}")
    pdf.drawString(50, 710, f"Confidence: {confidence}%")
    pdf.drawImage(os.path.join(app.config['UPLOAD_FOLDER'], image_path), 50, 500, width=200, height=200)

    pdf.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name='prediction_report.pdf', mimetype='application/pdf')

if __name__ == '__main__':
    try:
        app.run(debug=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user.")
