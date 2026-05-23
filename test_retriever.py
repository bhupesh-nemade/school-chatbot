from chatbot.retriever import get_retriever

retriever = get_retriever()

query = "What are the lunch items available in week 2?"

results = retriever.invoke(query)

print(f"Retrieved chunks: {len(results)}")

for i, doc in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print(doc.page_content[:500])
    print("\nMetadata:")
    print(doc.metadata)