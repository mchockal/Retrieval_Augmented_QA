# Retrieval_Augmented_QA
A Question-Answering API on a specific topic using RAG.
<div align = 'center'> 
	<img src ='https://github.com/mchockal/Retrieval_Augmented_QA/blob/main/data/Design_QA_With_RAG.png' alt="QA with RAG Design Diagram">
</div>

##1. Usage

There are three services in total, each in separate containers.
- weaviate : Local setup of vector store
- embedding_generator : To add any topic to knowledge base in its chunked + embedded format.
- qa_service : Main end-user facing service that takes user question and answers it based on context retrieved from vector store. 

To test, run the following commands from `app` directory

```bash
docker compose up -d 
```
After four minutes of waiting ( poor , but conscious design choices..still poor overall ), try the following:
- `POST request to localhost:8081/generate_and_save_embeddings` \
  -- Parses StreamingLLM.pdf, chunks it and stores text + embeddings in weaviate
- `GET request to localhost:8082/answer` \
  -- Default question "What is a KV cache" is used

Clean up after testing as follows:
```bash
docker compose down
```
- *Note : Make sure the bloated images are removed after testing.


