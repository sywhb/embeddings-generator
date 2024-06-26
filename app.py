import os
import time

from flask import Flask, request
from google.cloud import aiplatform
from langchain.embeddings import VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from markdownify import markdownify as md

app = Flask(__name__)
project_id = os.environ.get("PROJECT_ID")
region = os.environ.get("REGION")
model_name = 'textembedding-gecko'

# Generate the vector embeddings for each chunk of text.
# This code snippet may run for a few minutes.
aiplatform.init(project=f"{project_id}", location=f"{region}")
embeddings_service = VertexAIEmbeddings()


# Helper function to retry failed API requests with exponential backoff.
def retry_with_backoff(func, *args, retry_delay=5, backoff_factor=2, **kwargs):
    max_attempts = 10
    retries = 0
    for i in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"error: {e}")
            retries += 1
            wait = retry_delay * (backoff_factor**retries)
            print(f"Retry after waiting for {wait} seconds...")
            time.sleep(wait)


# convert html to markdown
def html_to_markdown(html):
    return md(html)


# Split long text descriptions into smaller chunks that can fit into
# the API request size limit, as expected by the LLM providers.
def split_text(text, is_html=False):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[".", "\n"],
        chunk_size=500,
        chunk_overlap=0,
        length_function=len,
    )
    chunked = []
    content = html_to_markdown(text) if is_html else text
    splits = text_splitter.create_documents([content])
    for s in splits:
        r = {"content": s.page_content}
        chunked.append(r)

    return chunked


# Generate the vector embeddings for each chunk of text.
# This code snippet may run for a few minutes.
def generate_embeddings(chunked):
    batch_size = 5
    for i in range(0, len(chunked), batch_size):
        request = [x["content"] for x in chunked[i: i + batch_size]]
        response = retry_with_backoff(embeddings_service.embed_documents, request)
        # Store the retrieved vector embeddings for each chunk back.
        for x, e in zip(chunked[i: i + batch_size], response):
            x["embedding"] = e

    return chunked[0]["embedding"]


def generate_embeded_query(query):
    return embeddings_service.embed_query(query)


def generate_embedded_text(text, is_html=False):
    chunked = split_text(text, is_html)
    return generate_embeddings(chunked)


@app.route("/", methods=["POST"])
def index():
    # get the request body
    request_body = request.get_json()
    text = request_body["text"]
    is_html = request_body["is_html"] if "is_html" in request_body else False
    is_query = request_body["is_query"] if "is_query" in request_body else False

    # get the embedded text
    embeddings = generate_embeded_query(
        text) if is_query else generate_embedded_text(text, is_html)

    # return the response 768 dimensional vector
    return {
        "embeddings": embeddings,
        "model": model_name
    }


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
