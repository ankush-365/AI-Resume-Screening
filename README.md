# 🤖 AI Resume Screening & Insights System

An AI-powered Resume Screening System that evaluates candidate resumes against a given Job Description using:

- 🔎 Semantic Similarity (Embeddings)
- 📚 Retrieval-Augmented Generation (RAG)
- 🧠 LLM-based Evaluation (Groq)
- 📊 Structured Feedback & Match Scoring
- 🌐 Streamlit Web Interface

---

## 🚀 Live Demo

Want to know if your resume is correctly fitted for the description: Try it now
https://ai-resume-screening-insights.streamlit.app/

---

## ⚙️ How It Works

### 1️⃣ Resume Upload and Job Description as Input
User uploads a PDF resume and gives a job description.

### 2️⃣ Text Extraction
PDF content is extracted using `pdfplumber`.

### 3️⃣ Chunking
Resume text is split into smaller chunks using: 'RecursiveCharacterTextSplitter'.

### 4️⃣ Embedding Generation
Embeddings are created using: HuggingFace sentence-transformer.

### 5️⃣ Vector Store
Chunks are stored in: ChromaDB.

### 6️⃣ Retrieval (RAG)
Top relevant resume sections are retrieved based on Job Description similarity.

### 7️⃣ Similarity Score
Cosine similarity between:
- Full Resume
- Job Description

Produces a percentage match score.

### 8️⃣ LLM Evaluation
Groq LLM analyzes:
- Matching Skills
- Missing Skills
- Strengths
- Weaknesses
- Final Recommendation

---

## 📊 Output Includes

-  Similarity Match Score
-  Matching Skills
-  Missing Skills
-  Strengths
-  Weaknesses
-  Final Recommendation (Strong / Moderate / Weak Fit)

