import os
from langchain_community.document_loaders import PyPDFLoader


PDF_FOLDER = "data/raw_pdfs"


def load_all_pdfs():
    documents = []

    for file_name in os.listdir(PDF_FOLDER):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(PDF_FOLDER, file_name)

            print(f"Loading: {file_name}")

            loader = PyPDFLoader(file_path)
            pdf_docs = loader.load()

            documents.extend(pdf_docs)

    return documents