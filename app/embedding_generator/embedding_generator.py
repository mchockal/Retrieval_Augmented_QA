"""
This file defines the endpoint for uploading or providing a file path to a PDF document.
It then parses the PDF document and optionally saves the parsed text to a file.

Endpoints:
- POST /upload-and-parse/: Accepts either an uploaded PDF file or a file path to a PDF.
It parses the PDF content and returns the parsed text as JSON.
Optionally, the `save` argument can be used to save the parsed text to a file in the "data" directory.

Usage:
- Default file (StreamingLLM.pdf) is always used, unless user provides one of the following two arguments
- To upload a PDF file, make a POST request to /upload-and-parse/ and include the PDF file (optional) in the request.
- To provide a file path to a PDF, make a POST request to /upload-and-parse/ and include the `file_path` (optional) parameter in the request body.

Arguments:
- file (UploadFile): An uploaded PDF file.
- file_path (str): A file path to a PDF document. Defaults to "data/StreamingLLM.pdf"
- save (optional, boolean, default: True): An argument that determines whether to save the parsed text to a file.

Responses:
- If successful, the endpoint returns the parsed text as JSON.

Error Handling:
- The endpoint provides error handling for cases where no file is provided, the uploaded file is not a PDF, or the file path is invalid.

"""
import json
import os

WEAVIATE_URL = "http://weaviate:8080"

from fastapi import FastAPI, File, UploadFile, HTTPException
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import weaviate

import pypdf
import io

app = FastAPI()


def parse_pdf(pdf_file):
    text = ""
    pdf_reader = pypdf.PdfReader(pdf_file)
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text


def save_text_to_file(text, file_path):
    with open(file_path, 'w', encoding="utf-8") as f:
        f.write(text)


def upload_and_parse(file=None, file_path: str = "./data/StreamingLLM.pdf", save=True) -> dict[str, str]:
    parsed_text = ""
    if file:
        if file.content_type != "application/pdf":
            return {"error": "The uploaded file is not a PDF"}
        pdf_content = file.file.read()
        pdf_file = io.BytesIO(pdf_content)
        parsed_text = parse_pdf(pdf_file)
    else:
        if not file_path.endswith(".pdf"):
            return {"error": "The provided file path is not a PDF"}
        try:
            with open(file_path, "rb") as pdf_file:
                parsed_text = parse_pdf(pdf_file)
        except FileNotFoundError:
            return {"error": "File not found"}

    if save:
        save_text_to_file(parsed_text, './data/parsed_text.txt')

    return {"file_name": './data/parsed_text.txt'}


def setup_schema(weaviate_client, schema_file="./data/weaviate_schema.json"):
    # Load the schema from the JSON file
    with open(schema_file, 'r') as json_file:
        class_schema = json.load(json_file)

    # Create the class in Weaviate
    response = weaviate_client.schema.get()
    response = response['classes']
    class_exists = any([cls['class'] == "Article" for cls in response])

    if class_exists:
        weaviate_client.schema.delete_class("Article")

    weaviate_client.schema.create_class(class_schema)


def check_batch_result(results: dict):
    """
  Ref: https://weaviate.io/developers/weaviate/client-libraries/python
  Check batch results for errors.

  Parameters
  ----------
  results : dict
      The Weaviate batch creation return value.
  """

    if results is not None:
        for result in results:
            if "result" in result and "errors" in result["result"]:
                if "error" in result["result"]["errors"]:
                    print(result["result"])


@app.get("/")
def home():
    return {
        "Usage:": "Upload a PDF file to use as your knowledge base for QA, or provide a valid file path to pdf file",
        "Post-URL": "/upload-and-parse/",
        "Eg": " curl -X POST -F \"file=@/path/to/your/file.pdf\" http://localhost:8000/upload-and-parse",
        "Default": "Uses data/StreamingLLM.pdf as knowledge base"}


@app.post("/generate-and-save-embeddings/")
async def generate_and_save_embeddings(file: UploadFile = File(None), file_path: str = "./data/StreamingLLM.pdf",
                                       chunk_size=500, save=True):
    try:
        retval = upload_and_parse(file_path=file_path)

        # Error in parsing user provided or default pdf file
        if retval.get("error"):
            raise HTTPException(status_code=400, detail=retval["error"])

        # Else, continue to chunk parsed_text.txt and save as embeddings in Weavite
        file_name = retval.get("file_name")
        loader = TextLoader(file_name, encoding="utf-8")
        document = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

        # Weaviate Setup
        client = weaviate.Client(os.environ.get('WEAVIATE_URL', WEAVIATE_URL))
        setup_schema(client)
        index = "article_index"
        client.batch.configure(
            batch_size=20,
            dynamic=True,
            creation_time=5,
            timeout_retries=3,
            connection_error_retries=3,
            callback=check_batch_result,
        )

        # Split the text into chunks
        chunked_doc = text_splitter.split_documents(document)
        chunked_embeddings = embeddings.embed_documents([doc.page_content for doc in chunked_doc])

        # Store embeddings in Weaviate
        with client.batch as batch:
            for i, doc in enumerate(chunked_doc):
                vector = chunked_embeddings[i]

                weaviate_object = {
                    "source_id": i,
                    "text": doc.page_content,
                }

                # Store the object in the specified Weaviate index using custom embeddings
                batch.add_data_object(data_object=weaviate_object,
                                      class_name="Article",
                                      vector=vector)

        return {"total_chunks": len(chunked_embeddings), "text": chunked_doc[0].page_content,
                "embedding": chunked_embeddings[0]}
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="File not found")
