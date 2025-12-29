import streamlit as st
import pandas as pd
import csv
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from langdetect import detect
from collections import Counter
import json

# ------------------------- Login System -------------------------

CREDENTIALS_FILE = "users.json"

def load_credentials():
    if not os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, "w") as f:
            json.dump({}, f)
    with open(CREDENTIALS_FILE, "r") as f:
        return json.load(f)

def save_credentials(credentials):
    with open(CREDENTIALS_FILE, "w") as f:
        json.dump(credentials, f, indent=4)


def login_page():
    st.set_page_config(page_title="Login - Parallel Corpus Analyzer", layout="centered")
    st.title("🔐 Welcome to Parallel Corpus Analyzer")

    option = st.radio("Choose Option", ["Login", "Sign Up"], horizontal=True)

    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")
    credentials = load_credentials()

    if option == "Login":
        if st.button("Login"):
            if username in credentials and credentials[username] == password:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.success("✅ Login successful!")
                st.rerun()

            else:
                st.error("❌ Invalid username or password")

    elif option == "Sign Up":
        confirm_password = st.text_input("🔁 Confirm Password", type="password")
        if st.button("Sign Up"):
            if username in credentials:
                st.error("⚠️ Username already exists. Try another.")
            elif password != confirm_password:
                st.error("❌ Passwords do not match.")
            elif username.strip() == "" or password.strip() == "":
                st.error("⚠️ Username and password cannot be empty.")
            else:
                credentials[username] = password
                save_credentials(credentials)
                st.success("✅ Registration successful! You can now log in.")
# ----------------- Authentication Check -----------------
if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    login_page()
    st.stop()
                
                
def detect_language(sentences, sample_size=20):
    sample_sentences = [s for s in sentences if s.strip()]
    sample_sentences = sample_sentences[:sample_size]  # Use first 20 non-empty sentences

    predictions = []
    for text in sample_sentences:
        try:
            predictions.append(detect(text))
        except:
            continue

    if not predictions:
        return "unknown"

    # Return most common language code
    return Counter(predictions).most_common(1)[0][0]
                

# ------------------------- Define Analysis Functions -------------------------

def run_semantic_analysis(df):
    os.makedirs("output", exist_ok=True)
    df.columns = [col.strip().lower() for col in df.columns]
    if 'source' not in df.columns or 'target' not in df.columns:
        raise ValueError("CSV must contain 'Source' and 'Target' columns.")

    df.rename(columns={"source": "English", "target": "Hindi"}, inplace=True)
    df = df[['English', 'Hindi']]
    df['English'] = df['English'].astype(str).str.strip().str.lower()
    df['Hindi'] = df['Hindi'].astype(str).str.strip()

    model = SentenceTransformer('sentence-transformers/LaBSE')
    english_embeddings = model.encode(df['English'].tolist(), convert_to_tensor=True)
    hindi_embeddings = model.encode(df['Hindi'].tolist(), convert_to_tensor=True)

    def find_duplicates(embeddings, sentences, threshold=0.95):
        similarity_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
        duplicates = []
        visited = set()
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix[i][j] >= threshold and (i, j) not in visited:
                    duplicates.append((i, j, similarity_matrix[i][j], sentences[i], sentences[j]))
                    visited.add((i, j))
        return duplicates

    eng_dupes = find_duplicates(english_embeddings, df['English'].tolist())
    hin_dupes = find_duplicates(hindi_embeddings, df['Hindi'].tolist())

    def save_duplicate_report(duplicates, filename, lang):
        with open(filename, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"{lang} Index 1", f"{lang} Index 2", "Similarity", f"{lang} Sentence 1", f"{lang} Sentence 2"])
            for i, j, sim, s1, s2 in duplicates:
                writer.writerow([i, j, round(sim, 4), s1, s2])

    save_duplicate_report(eng_dupes, "output/english_duplicates_labse.csv", "English")
    save_duplicate_report(hin_dupes, "output/hindi_duplicates_labse.csv", "Hindi")

    similarity_scores = util.cos_sim(english_embeddings, hindi_embeddings).diagonal()
    df['Similarity'] = similarity_scores.cpu().numpy()

    def classify_alignment(score):
        if score >= 0.70:
            return "Aligned"
        else:
            return "Misaligned"

    df['Alignment Quality'] = df['Similarity'].apply(classify_alignment)
    df.to_csv("output/corpus_with_similarity_labeled.csv", index=False, encoding='utf-8')

    plt.figure(figsize=(10, 5))
    sns.histplot(df['Similarity'], bins=20, kde=True)
    plt.title("Sentence Similarity Distribution (LaBSE)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("output/sentence_similarity_distribution_plot.png")
    plt.close()

    semantic_src_dup_ids = set([i for i, j, _, _, _ in eng_dupes] + [j for i, j, _, _, _ in eng_dupes])
    semantic_tgt_dup_ids = set([i for i, j, _, _, _ in hin_dupes] + [j for i, j, _, _, _ in hin_dupes])

    return {
    "Aligned Sentence Pairs": (df['Alignment Quality'] == "Aligned").sum(),
    "Misaligned Sentence Pairs": (df['Alignment Quality'] == "Misaligned").sum(),
    "Semantic Source Duplicates": len(semantic_src_dup_ids),
    "Semantic Target Duplicates": len(semantic_tgt_dup_ids)
  }

def run_statistical_analysis(df):
    source_col = "Source"
    target_col = "Target"
    df.columns = [source_col, target_col]
    df[source_col] = df[source_col].astype(str).str.strip()
    df[target_col] = df[target_col].astype(str).str.strip()

    def count_real_words(text):
        return len([w for w in text.split() if re.search(r"\w", w)])

    source_lengths = df[source_col].apply(count_real_words).tolist()
    target_lengths = df[target_col].apply(count_real_words).tolist()

    rows = []
    for i, (src, tgt, sl, tl) in enumerate(zip(df[source_col], df[target_col], source_lengths, target_lengths), 1):
        ratio = round(tl / sl, 2) if sl > 0 else 0
        alignment = "Misaligned" if ratio == 0 or ratio < 0.5 or ratio > 2.0 else "Aligned"
        rows.append([i, src, tgt, sl, tl, ratio, alignment])

    with open("output/sentence_length_stats.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Source", "Target", "Source Length", "Target Length", "Length Ratio", "Alignment Quality"])
        writer.writerows(rows)

    length_map = defaultdict(list)
    for sl, tl in zip(source_lengths, target_lengths):
        if sl > 0:
            length_map[sl].append(tl)

    avg_target_per_src_len = {k: round(sum(v)/len(v), 2) for k, v in length_map.items()}
    avg_ratio = round(sum(target_lengths) / sum(source_lengths), 2) if sum(source_lengths) > 0 else 0

    with open("output/length_ratio_by_source_length.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Source Length", "Avg Target Length"])
        for k in sorted(avg_target_per_src_len):
            writer.writerow([k, avg_target_per_src_len[k]])

    def normalize_text(text):
        text = str(text).strip().lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text

    def save_duplicates_with_indices(sentences, filename):
        index_map = defaultdict(list)
        for idx, sent in enumerate(sentences):
            cleaned = normalize_text(sent)
            index_map[cleaned].append(idx)
        duplicates = [(sent, len(idxs), idxs) for sent, idxs in index_map.items() if len(idxs) > 1]
        with open(filename, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Normalized Sentence", "Count", "Indices"])
            for sent, count, idxs in duplicates:
                writer.writerow([sent, count, ', '.join(map(str, idxs))])

    save_duplicates_with_indices(df[source_col], "output/source_duplicates.csv")
    save_duplicates_with_indices(df[target_col], "output/target_duplicates.csv")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(source_lengths, bins=15, color='skyblue', edgecolor='black')
    plt.title("Histogram of Source Sentence Lengths")
    plt.xlabel("Length (words)")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(target_lengths, bins=15, color='salmon', edgecolor='black')
    plt.title("Histogram of Target Sentence Lengths")
    plt.xlabel("Length (words)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("output/sentence_length_histograms_plot.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(source_lengths, target_lengths, alpha=0.6, color='purple', edgecolors='black')
    plt.title("Scatter Plot: Source vs Target Sentence Lengths")
    plt.xlabel("Source Length (words)")
    plt.ylabel("Target Length (words)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output/scatter_plot_source_vs_target.png")
    plt.close()
    
    def get_exact_duplicates_count(sentences):
       index_map = defaultdict(list)
       for idx, sent in enumerate(sentences):
           cleaned = normalize_text(sent)
           index_map[cleaned].append(idx)
    # count unique sentence indices that appear more than once
       dup_indices = set()
       for idxs in index_map.values():
          if len(idxs) > 1:
            dup_indices.update(idxs)
       return len(dup_indices)

    src_dup_count = get_exact_duplicates_count(df[source_col])
    tgt_dup_count = get_exact_duplicates_count(df[target_col])


    return {
    "Total Sentence Pairs": len(df),
    "Max Source Length": max(source_lengths),
    "Min Source Length": min(source_lengths),
    "Max Target Length": max(target_lengths),
    "Min Target Length": min(target_lengths),
    "Average Target/Source Length Ratio": avg_ratio,
    "Exact Source Duplicates": src_dup_count,
    "Exact Target Duplicates": tgt_dup_count
}
# ------------------------- Main Streamlit App -------------------------

# ------------------------- Main Streamlit App -------------------------

st.set_page_config(page_title="Parallel Corpus Analyzer", layout="wide")
st.title("📚 Parallel Corpus Text Analysis Tool")

with st.sidebar:
    st.header("📥 Upload Your Corpus File")
    uploaded_file = st.file_uploader("Upload CSV with at least 2 columns (header or no header)", type="csv")
    st.markdown("---")
    st.markdown(f"👤 Logged in as: **{st.session_state['username']}**")
    if st.button("🚪 Logout"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = "" 
        st.rerun()
    st.markdown("📌 **Instructions:**\n"
                "- Upload a CSV with at least two columns\n"
                "- Click **Run Analysis** to start\n"
                "- View summary, charts, and download results\n")
    st.markdown("🔍 **Note:** The app uses LaBSE for semantic similarity.")

st.markdown("""
This tool performs:
- ✨ **Semantic similarity** using LaBSE
- 📉 **Length-based analysis** (word count, ratio)
- 🔁 **Duplicate detection**
- 📊 **Interactive statistics & charts**
- 📥 **Downloadable CSV reports**
""")

if uploaded_file:
    try:
        # Try reading the file with header first
        df = pd.read_csv(uploaded_file)
        if df.shape[1] < 2:
            raise ValueError("File has fewer than two columns.")
        original_columns = df.columns.tolist()
    except Exception:
        # If header reading fails (or file has no header), fallback to no header
        df = pd.read_csv(uploaded_file, header=None)
        if df.shape[1] < 2:
            st.error("❌ The uploaded file must contain at least two columns.")
            st.stop()
        original_columns = ["Column 1", "Column 2"]

    df = df.iloc[:, :2]
    df.columns = ['Source', 'Target']

    st.info(f"🔍 Using first two columns: `{original_columns[0]}` and `{original_columns[1]}` as `Source` and `Target`.")
    st.success("✅ File uploaded and columns renamed successfully!")

    try:
        # Detect languages
        src_lang = detect_language(df['Source'].dropna().astype(str).tolist())
        tgt_lang = detect_language(df['Target'].dropna().astype(str).tolist())

        lang_map = {
            'en': 'English',
            'hi': 'Hindi',
            'fr': 'French',
            'de': 'German',
            'es': 'Spanish',
            'zh-cn': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ru': 'Russian',
            'it': 'Italian',
            'pt': 'Portuguese',
            'unknown': 'Unknown'
        }

        src_lang_full = lang_map.get(src_lang, src_lang)
        tgt_lang_full = lang_map.get(tgt_lang, tgt_lang)

        st.markdown("### 🌍 Detected Languages")
        st.markdown(f"- **Source Column**: `{src_lang_full}`")
        st.markdown(f"- **Target Column**: `{tgt_lang_full}`")
    except Exception as e:
        st.warning(f"⚠️ Language detection failed: {e}")

    with st.expander("🔍 Preview Uploaded Data", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    if st.button("🚀 Run Analysis"):
        with st.spinner("Running LaBSE semantic and statistical analysis..."):
            sem_stats = run_semantic_analysis(df.copy())
            stat_stats = run_statistical_analysis(df.copy())

        st.success("✅ Analysis Complete!")
        st.subheader("📌 Summary Statistics")
        summary = {**stat_stats, **sem_stats}
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        for i, (key, value) in enumerate(summary.items()):
            cols[i % 3].metric(label=key, value=value)

        st.markdown("## 📈 Semantic Similarity Distribution")
        st.image("output/sentence_similarity_distribution_plot.png", caption="Distribution of Semantic Similarity Scores")

        with st.expander("📥 Download Semantic Analysis Results"):
            st.download_button("📄 Similarity Labeled Corpus", open("output/corpus_with_similarity_labeled.csv", "rb"), "corpus_with_similarity_labeled.csv")
            st.download_button("📄 Semantic Source Duplicates", open("output/english_duplicates_labse.csv", "rb"), "english_duplicates_labse.csv")
            st.download_button("📄 Semantic Target Duplicates", open("output/hindi_duplicates_labse.csv", "rb"), "hindi_duplicates_labse.csv")

        st.markdown("## 📊 Sentence Length Analysis")
        st.image("output/sentence_length_histograms_plot.png", caption="Histograms: Source vs Target Sentence Lengths")
        st.image("output/scatter_plot_source_vs_target.png", caption="Scatter: Source vs Target Sentence Lengths")

        with st.expander("📥 Download Statistical Reports"):
            st.download_button("📄 Sentence Length Stats", open("output/sentence_length_stats.csv", "rb"), "sentence_length_stats.csv")
            st.download_button("📄 Avg Target per Source Length", open("output/length_ratio_by_source_length.csv", "rb"), "length_ratio_by_source_length.csv")
            st.download_button("📄 Exact Source Duplicates", open("output/source_duplicates.csv", "rb"), "source_duplicates.csv")
            st.download_button("📄 Exact Target Duplicates", open("output/target_duplicates.csv", "rb"), "target_duplicates.csv")
else:
    st.info("📂 Please upload a CSV file from the **sidebar** to get started.")
