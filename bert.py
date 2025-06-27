from transformers import pipeline
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

# Load pre-trained extractive QA model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Load and chunk the context
def load_and_chunk_context(file_path, sentences_per_chunk=5):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    sentences = sent_tokenize(text)
    chunks = [
        ' '.join(sentences[i:i+sentences_per_chunk])
        for i in range(0, len(sentences), sentences_per_chunk)
    ]
    return chunks

# Run QA on each chunk and return best answer
def ask_question_across_chunks(chunks, question, threshold=0.5):
    best_answer = {"answer": "", "score": 0.0}
    
    for chunk in chunks:
        result = qa_pipeline(question=question, context=chunk)
        if result["score"] > best_answer["score"]:
            best_answer = result
            best_answer["context"] = chunk  # optional: track source
    
    if best_answer["score"] >= threshold:
        return f"A: {best_answer['answer']} (confidence: {best_answer['score']*100:.1f}%)"
    else:
        return "⚠️ No reliable answer found (confidence below threshold)."

# Main interactive loop
if __name__ == "__main__":
    context_file = "context.txt"
    try:
        chunks = load_and_chunk_context(context_file)
        print(f"✅ Context loaded and split into {len(chunks)} chunks.\n")
        print("Ask a question (type 'exit' to quit):\n")

        while True:
            question = input("Q: ")
            if question.lower() in ['exit', 'quit']:
                break
            answer = ask_question_across_chunks(chunks, question, threshold=0.5)
            print(answer + "\n")

    except FileNotFoundError:
        print(f"❌ Error: '{context_file}' not found.")
