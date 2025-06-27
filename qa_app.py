import streamlit as st
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize


nltk.download("punkt")

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def chunk_context(text, sentences_per_chunk=5):
    sentences = sent_tokenize(text)
    chunks = [
        " ".join(sentences[i:i+sentences_per_chunk])
        for i in range(0, len(sentences), sentences_per_chunk)
    ]
    return chunks

def get_best_answer(question, chunks, threshold=0.2):
    best = {"answer": "", "score": 0.0}
    for chunk in chunks:
        result = qa_pipeline(question=question, context=chunk)
        if result["score"] > best["score"]:
            best = result   
            best["context"] = chunk
    return best if best["score"] >= threshold else None

# Streamlit UI
st.title("🧠 BERT Extractive QA")
st.write("Upload a text file and ask a question about its content.")

uploaded_file = st.file_uploader("Upload context.txt", type="txt")
question = st.text_input("Your Question")

if uploaded_file:
    context = uploaded_file.read().decode("utf-8")
    chunks = chunk_context(context)

    if question:
        result = get_best_answer(question, chunks)
        if result:
            st.success(f"**Answer**: {result['answer']}")
            st.write(f"Confidence: {result['score']*100:.1f}%")
        else:
            st.warning("🤔 No reliable answer found (confidence below threshold).")
