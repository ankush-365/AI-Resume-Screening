import streamlit as st
import pdfplumber
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("🤖 AI Resume Screening & Feedback System")

# ---------------------------
# SIDEBAR - API KEY INPUT
# ---------------------------

groq_api_key = st.secrets["GROQ_API_KEY"]

# ---------------------------
# MAIN INPUTS
# ---------------------------
jd = st.text_area("📄 Paste Job Description Here", height=250)
resume_file = st.file_uploader("📎 Upload Resume (PDF)", type=["pdf"])

# ---------------------------
# PROCESS BUTTON
# ---------------------------
if st.button("🚀 Evaluate Resume"):

    if not groq_api_key:
        st.error("Please enter your Groq API Key in the sidebar.")
        st.stop()

    if not jd.strip():
        st.error("Please paste a Job Description.")
        st.stop()

    if not resume_file:
        st.error("Please upload a resume.")
        st.stop()

    # ---------------------------
    # Extract Resume Text
    # ---------------------------
    with pdfplumber.open(resume_file) as pdf:
        full_resume_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_resume_text += text + "\n"

    if not full_resume_text.strip():
        st.error("Could not extract text from PDF.")
        st.stop()

    # ---------------------------
    # Chunking
    # ---------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    docs = text_splitter.split_documents(
        [Document(page_content=full_resume_text)]
    )

    # ---------------------------
    # Embeddings
    # ---------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(jd)
    retrieved_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # ---------------------------
    # Similarity Score
    # ---------------------------
    jd_embedding = embeddings.embed_query(jd)
    resume_embedding = embeddings.embed_query(full_resume_text)

    similarity = cosine_similarity(
        [jd_embedding],
        [resume_embedding]
    )[0][0]

    score = round(float(similarity) * 100, 2)

    # ---------------------------
    # LLM Setup
    # ---------------------------
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model="openai/gpt-oss-120b"
    )

    prompt = f"""
You are an AI Resume Screening System.

Embedding Similarity Score: {score}%

Job Description:
{jd}

Candidate Resume (Relevant Sections):
{retrieved_text}

Evaluate strictly and return:

1. Final Match Score (0-100)
2. Matching Skills
3. Missing Skills
4. Strengths
5. Weaknesses
6. Final Recommendation (Strong Fit / Moderate Fit / Weak Fit)
"""

    response = llm.invoke(prompt)

    # ---------------------------
    # Fit Label Logic
    # ---------------------------
    # if score >= 75:
    #     fit = "Strong Fit"
    #     fit_display = st.success
    # elif score >= 45:
    #     fit = "Moderate Fit"
    #     fit_display = st.warning
    # else:
    #     fit = "Weak Fit"
    #     fit_display = st.error

    # ---------------------------
    # DISPLAY RESULTS
    # ---------------------------

    st.divider()
    st.header("📝 AI Detailed Evaluation")
    st.write(response.content)
