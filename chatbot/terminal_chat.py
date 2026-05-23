# from chatbot.chain import ask_question


# def main():
#     print("=" * 60)
#     print("School Chatbot Started")
#     print("Type 'exit' to quit")
#     print("=" * 60)

#     while True:
#         question = input("\nYou: ").strip()

#         if question.lower() in ["exit", "quit"]:
#             print("Chatbot stopped.")
#             break

#         if not question:
#             print("Please enter a question.")
#             continue

#         try:
#             answer, docs = ask_question(question)

#             print("\nBot:")
#             print(answer)

#             print("\nSources:")
#             for doc in docs:
#                 source = doc.metadata.get("source", "Unknown")
#                 page = doc.metadata.get("page", "Unknown")
#                 print(f"- {source} | Page: {page}")

#         except Exception as e:
#             print(f"\nError: {e}")


# if __name__ == "__main__":
#     main()



from chatbot.chain import ask_question
from halo import Halo


def main():
    chat_history = []

    print("=" * 60)
    print("School Chatbot Started")
    print("Type 'exit' to quit")
    print("=" * 60)

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Chatbot stopped.")
            break

        if not question:
            print("Please enter a question.")
            continue

        spinner = Halo(text="Processing your question...", spinner="dot")

        try:
            spinner.start()

            answer, docs = ask_question(question, chat_history)

            spinner.stop()

            print("\nBot:")
            print(answer)

            print("\nSources:")
            for doc in docs:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "Unknown")
                print(f"- {source} | Page: {page}")

            chat_history.append((question, answer))

        except Exception as e:
            spinner.stop()
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()