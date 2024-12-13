import os
import pdfplumber
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Directory containing PDFs
PDF_DIRECTORY = "pdfs"

# Extract text from PDFs
def preprocess_pdfs(pdf_directory):
    all_texts = []
    for pdf_file in os.listdir(pdf_directory):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
                all_texts.append({"filename": pdf_file, "content": text})
    return all_texts

# Create and save vector store
def create_and_save_vectorstore(text_data, embedding_model='text-embedding-ada-002', store_path='f150_vectorstore'):
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectorstore = FAISS.from_texts([entry["content"] for entry in text_data], embeddings)
    vectorstore.save_local(store_path)

if __name__ == "__main__":
    text_data = preprocess_pdfs(PDF_DIRECTORY)
    create_and_save_vectorstore(text_data)