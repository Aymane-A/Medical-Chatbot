from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from src.prompt import *
import os

from store_index import PINECONE_API_KEY

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

embeddings = download_embeddings()
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings, 
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

ChatModel = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}")
    ]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": Retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatModel
    | StrOutputParser()
)



@app.route('/get', methods=["GET", "POST"])
def chat():
    msg = request.form['msg']
    response = rag_chain.invoke(msg)
    return str(response)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)