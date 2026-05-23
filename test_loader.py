from ingestion.load_pdfs import load_all_pdfs

docs = load_all_pdfs()

for i, doc in enumerate(docs):
    if "Week 2" in doc.page_content or "Tuesday" in doc.page_content:
        print(f"\n--- PAGE {i+1} ---")
        print(doc.page_content[:4000])