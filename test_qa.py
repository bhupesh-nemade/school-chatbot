from chatbot.chain import ask_question

answer, docs = ask_question("What is the school location?")

print("\nANSWER:\n")
print(answer)

print("\nSOURCES USED:\n")
for doc in docs:
    print(doc.metadata)