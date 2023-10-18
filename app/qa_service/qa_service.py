from functools import lru_cache

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from fastapi import FastAPI, HTTPException
from langchain.vectorstores import Weaviate
from pathlib import Path
import weaviate
import os

WEAVIATE_URL = os.environ.get('WEAVIATE_URL', "http://weaviate:8080")
os.environ['OPENAI_API_KEY'] = "INSERT_OPEAI_API_KEY"
app = FastAPI()

@lru_cache(maxsize=None)
def init():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    prompt = PromptTemplate(
        input_variables=["context", "user_query"],
        template=Path("./data/prompts/augment_user_query.prompt").read_text(),
    )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
    weaviate_client = weaviate.Client(WEAVIATE_URL)
    vectordb = Weaviate(weaviate_client, text_key="text", by_text=False, index_name="Article", embedding=embeddings)
    chain = RetrievalQA.from_chain_type(llm,
                                        retriever=vectordb.as_retriever(),
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": prompt})

    return chain, vectordb, embeddings


def answer_user_query(user_query, chain, vectordb, embeddings):
    query_embedding = embeddings.embed_query(user_query)
    context = vectordb.similarity_search_by_vector(query_embedding,k=3)
    result = chain({"query":user_query, "context":context})
    return result


@app.get("/answer")
def get_answer(user_query: str = "What is a KV cache?"):
    chain, vectordb, embeddings = init()
    try:
        user_answer = answer_user_query(user_query, chain, vectordb, embeddings)
        return {"answer": user_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error while processing the request"+str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)