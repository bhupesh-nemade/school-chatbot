from langchain_huggingface import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)

vec = emb.embed_query("hello world")

print(len(vec))