from functools import lru_cache
import time

from langchain_openai import ChatOpenAI
from chatbot.retriever import get_retriever, get_vectorstore
from config import OPENROUTER_API_KEY


@lru_cache(maxsize=1)
def get_llm():
    return ChatOpenAI(
        model="openai/gpt-oss-120b:free",
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0
    )


def format_docs(docs):
    formatted_docs = []

    for index, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "Unknown page")

        formatted_docs.append(
            f"[Document {index} | Source: {source} | Page: {page}]\n"
            f"{doc.page_content[:1200]}"
        )

    return "\n\n".join(formatted_docs)


def rewrite_question(question, chat_history):
    if not chat_history:
        return question

    lower_question = question.lower().strip()

    followup_words = [
        "what about",
        "and",
        "also",
        "that",
        "it",
        "those",
        "them",
        "this"
    ]

    if any(lower_question.startswith(word) for word in followup_words):
        last_user_question = chat_history[-1][0]
        return f"{last_user_question} {question}"

    return question



def ask_question(question, chat_history=None):
    if chat_history is None:
        chat_history = []

    start = time.time()

   

  
    vectorstore = get_vectorstore()
    llm = get_llm()

    standalone_question = rewrite_question(question, chat_history)

    print(f"Rewrite time: {time.time() - start:.2f} sec")

    retrieval_start = time.time()

    results = vectorstore.similarity_search_with_score(
        standalone_question,
        k=3
    )

    print(f"Initial retrieval time: {time.time() - retrieval_start:.2f} sec")

    if not results:
        return "I do not have information related to your question.", []

    best_score = results[0][1]

    print(f"Best similarity score: {best_score}")
    print(f"Question: {standalone_question}")

    if best_score > 1.0:
        return "I do not have information related to your question.", []

    docs = [doc for doc, score in results]

    expanded_docs = list(docs)
    seen_pages = set()

    for doc in docs:
        source = doc.metadata.get("source")
        page = doc.metadata.get("page")

        if source is None or page is None:
            continue

        seen_pages.add((source, page))

    expansion_start = time.time()

    if docs:
        top_doc = docs[0]

        source = top_doc.metadata.get("source")
        page = top_doc.metadata.get("page")

        if source is not None and page is not None:
            neighbor_pages = [page - 1, page + 1]

            for neighbor in neighbor_pages:
                if neighbor < 0:
                    continue

                neighbor_docs = vectorstore.similarity_search(
                    "",
                    k=1,
                    filter={
                        "source": source,
                        "page": neighbor
                    }
                )

                for ndoc in neighbor_docs:
                    if (source, neighbor) not in seen_pages:
                        expanded_docs.append(ndoc)
                        seen_pages.add((source, neighbor))

    print(f"Neighbor expansion time: {time.time() - expansion_start:.2f} sec")

    context = format_docs(expanded_docs)

    prompt = f"""
You are a school document assistant.

Your only source of truth is the retrieved PDF context below.

Strict rules:
- Answer ONLY from the retrieved PDF context.
- Every factual statement must be directly supported by the retrieved context.
- Do NOT use general knowledge.
- Do NOT guess.
- Do NOT infer missing details.
- If context is insufficient, say exactly:
  "I do not have information related to your question."
- Never create facts that are not explicitly present.

Retrieved PDF Context:
{context if context else "No relevant PDF context was retrieved."}

Question:
{standalone_question}
"""

    llm_start = time.time()
    response = llm.invoke(prompt)

    print(f"LLM response time: {time.time() - llm_start:.2f} sec")
    print(f"TOTAL TIME: {time.time() - start:.2f} sec")

    return response.content, expanded_docs