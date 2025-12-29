from flask import Flask, render_template, request, send_file, jsonify, url_for
import os
import pandas as pd
import re
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from collections import Counter
import seaborn as sns
import uuid
import torch
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

df_store = {}
df_cleaned_store = {}
progress_tracker = {}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('sentence-transformers/LaBSE', device=device)

def clean_text_advanced(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[^;\s]+;', '', text)
    text = re.sub(r'\bDrawingGrid.*?\b', '', text)
    text = re.sub(r'\bw:[^>\s]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def is_english_only(text):
    if not isinstance(text, str): return False
    return re.fullmatch(r"[a-zA-Z0-9\s.,!?;:'\"()\-/%&]+", text.strip()) is not None

def is_valid_sentence(text):
    if not isinstance(text, str): return False
    text = text.strip()
    if len(text.split()) < 2: return False
    if re.match(r"^[^a-zA-Z0-9]+$", text): return False
    if all(len(word) <= 2 for word in text.split()): return False
    if re.search(r'[<>]|&[a-z]+;', text): return False
    return True

def compute_stats(df):
    df['eng_len'] = df['COLUMN-1'].astype(str).apply(lambda x: len(x.split()))
    df['hin_len'] = df['COLUMN-2'].astype(str).apply(lambda x: len(x.split()))
    data = df[['COLUMN-1', 'COLUMN-2']].copy()
    if 'SIMILARITY' in df.columns:
        data['SIMILARITY'] = df['SIMILARITY']
    return {
        'total_sentences': len(df),
        'max_eng_len': df['eng_len'].max(),
        'avg_eng_len': round(df['eng_len'].mean(), 2),
        'std_eng_len': round(df['eng_len'].std(), 2),
        'max_hin_len': df['hin_len'].max(),
        'avg_hin_len': round(df['hin_len'].mean(), 2),
        'std_hin_len': round(df['hin_len'].std(), 2),
        'data': data.to_dict(orient='records')
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if not file: return 'No file uploaded', 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    df = pd.read_csv(filepath, encoding='utf-8')
    if not {'COLUMN-1', 'COLUMN-2'}.issubset(df.columns):
        return 'CSV must contain COLUMN-1 and COLUMN-2', 400

    df_store[filename] = df.copy()
    df_cleaned_store[filename] = df.copy()
    stats = compute_stats(df)
    return render_template('components/stats.html', stats=stats, filename=filename, df=df)

@app.route('/operation/<op_type>', methods=['POST'])
def operate(op_type):
    filename = request.form['filename']
    df = df_cleaned_store.get(filename)
    if df is None: return 'No file loaded', 400

    if op_type == 'remove_html':
        df['COLUMN-1'] = df['COLUMN-1'].astype(str).apply(clean_text_advanced)
        df['COLUMN-2'] = df['COLUMN-2'].astype(str).apply(clean_text_advanced)
    elif op_type == 'remove_short':
        df = df[df['COLUMN-1'].str.split().str.len() >= 2]
        df = df[df['COLUMN-2'].str.split().str.len() >= 2]
    elif op_type == 'remove_identical':
        df = df[df['COLUMN-1'] != df['COLUMN-2']]
    elif op_type == 'remove_english_in_hindi':
        df = df[~df['COLUMN-2'].apply(is_english_only)]
    elif op_type == 'remove_invalid':
        df = df[df['COLUMN-1'].apply(is_valid_sentence) & df['COLUMN-2'].apply(is_valid_sentence)]
    elif op_type == 'remove_duplicate_pairs':
        df = df.drop_duplicates(subset=['COLUMN-1', 'COLUMN-2'])
    elif op_type == 'remove_duplicate_column1':
        df = df.drop_duplicates(subset=['COLUMN-1'])
    elif op_type == 'remove_duplicate_column2':
        df = df.drop_duplicates(subset=['COLUMN-2'])
    elif op_type == 'full_clean':
        df['COLUMN-1'] = df['COLUMN-1'].astype(str).apply(clean_text_advanced)
        df['COLUMN-2'] = df['COLUMN-2'].astype(str).apply(clean_text_advanced)
        df = df[df['COLUMN-1'].str.split().str.len() >= 2]
        df = df[df['COLUMN-2'].str.split().str.len() >= 2]
        df = df[df['COLUMN-1'] != df['COLUMN-2']]
        df = df[~df['COLUMN-2'].apply(is_english_only)]
        df = df[df['COLUMN-1'].apply(is_valid_sentence) & df['COLUMN-2'].apply(is_valid_sentence)]
        df = df.drop_duplicates(subset=['COLUMN-1', 'COLUMN-2'])
        df = df.drop_duplicates(subset=['COLUMN-1'])
        df = df.drop_duplicates(subset=['COLUMN-2'])

    df_cleaned_store[filename] = df.copy()
    cleaned_path = os.path.join(UPLOAD_FOLDER, f"cleaned_{filename}")
    df.to_csv(cleaned_path, index=False, encoding='utf-8-sig')
    stats = compute_stats(df)
    return render_template('components/stats.html', stats=stats, filename=filename, df=df)

@app.route('/analyze/<analytic_type>', methods=['POST'])
def analyze(analytic_type):
    filename = request.form['filename']
    df = df_cleaned_store.get(filename)
    if df is None: return 'No data loaded', 400

    df['eng_len'] = df['COLUMN-1'].astype(str).apply(lambda x: len(x.split()))
    df['hin_len'] = df['COLUMN-2'].astype(str).apply(lambda x: len(x.split()))
    plt.figure(figsize=(8, 5))

    if analytic_type == 'length_distribution':
        sns.histplot(df['eng_len'], color='blue', label='English', kde=True)
        sns.histplot(df['hin_len'], color='green', label='Hindi', kde=True)
        plt.title('Sentence Length Distribution')
        plt.legend()
    elif analytic_type == 'word_count_scatter':
        plt.scatter(df['eng_len'], df['hin_len'], alpha=0.5, color='purple')
        plt.xlabel('English Word Count')
        plt.ylabel('Hindi Word Count')
        plt.title('Word Count Scatter Plot')
    elif analytic_type == 'length_diff_hist':
        diff = (df['eng_len'] - df['hin_len']).abs()
        sns.histplot(diff, bins=20, color='orange')
        plt.title('Length Difference Histogram')
    elif analytic_type == 'duplicates':
        dup1 = df.duplicated(subset=['COLUMN-1']).sum()
        dup2 = df.duplicated(subset=['COLUMN-2']).sum()
        plt.bar(['Eng Duplicates', 'Hin Duplicates'], [dup1, dup2], color=['red', 'green'])
        plt.title('Duplicate Sentences Count')
    elif analytic_type == 'top_words':
        words = ' '.join(df['COLUMN-1'].astype(str)).lower().split()
        top = Counter(words).most_common(10)
        labels, values = zip(*top)
        plt.bar(labels, values, color='teal')
        plt.title('Top 10 Frequent English Words')
        plt.xticks(rotation=45)
    elif analytic_type == 'extreme_sentences':
        idx_min = df['eng_len'].idxmin()
        idx_max = df['eng_len'].idxmax()
        min_sent = df.loc[idx_min, 'COLUMN-1']
        max_sent = df.loc[idx_max, 'COLUMN-1']
        labels = ['Shortest', 'Longest']
        values = [len(min_sent.split()), len(max_sent.split())]
        plt.bar(labels, values, color=['gray', 'navy'])
        plt.title('Shortest vs Longest English Sentences')
    else:
        return 'Unknown analytic type', 400

    unique_id = uuid.uuid4().hex
    chart_filename = f"analytics_{analytic_type}_{filename}_{unique_id}.png"
    chart_path = os.path.join(STATIC_FOLDER, chart_filename)
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    return jsonify({'chart': url_for('static', filename=chart_filename)})

@app.route('/similarity', methods=['POST'])
def similarity():
    filename = request.form['filename']
    df = df_cleaned_store.get(filename)
    if df is None:
        return jsonify({'error': 'No file loaded'}), 400

    BATCH_SIZE = 10
    total = len(df)
    similarities = []

    for i in range(0, total, BATCH_SIZE):
        print(i)
        src = df['COLUMN-1'].iloc[i:i + BATCH_SIZE].tolist()
        trg = df['COLUMN-2'].iloc[i:i + BATCH_SIZE].tolist()

        src_emb = model.encode(src, convert_to_tensor=True, device=device)
        trg_emb = model.encode(trg, convert_to_tensor=True, device=device)

        sim_scores = util.cos_sim(src_emb, trg_emb).diagonal().tolist()
        similarities.extend(sim_scores)

        progress_tracker[filename] = f"{min(i + BATCH_SIZE, total)} / {total}"

    df['SIMILARITY'] = similarities
    df_cleaned_store[filename] = df.copy()
    path = os.path.join(UPLOAD_FOLDER, f"cleaned_{filename}")
    df.to_csv(path, index=False, encoding='utf-8-sig')
    progress_tracker[filename] = f"Done: {total} / {total}"

    return jsonify({'status': 'completed'})

@app.route('/progress/<filename>')
def progress(filename):
    return jsonify({'progress': progress_tracker.get(filename, '0 / ?')})

@app.route('/download/<filename>')
def download(filename):
    path = os.path.join(UPLOAD_FOLDER, f"cleaned_{filename}")
    return send_file(path, as_attachment=True)

@app.route('/operation/refresh', methods=['POST'])
def refresh():
    filename = request.form['filename']
    df = df_cleaned_store.get(filename)
    if df is None:
        return 'No file loaded', 400
    stats = compute_stats(df)
    return render_template('components/stats.html', stats=stats, filename=filename, df=df)

@app.route('/developers')
def developers():
    return render_template('developer.html')

if __name__ == '__main__':
    app.run(debug=True)
