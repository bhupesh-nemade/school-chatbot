from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=500,
        separators=[
          "\n\n",
          "\n",
          ".",
          " ",
          ""
        ]
    )

    chunks = splitter.split_documents(documents)

    return chunks