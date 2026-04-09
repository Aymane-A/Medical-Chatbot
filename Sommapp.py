from fileinput import filename
import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()






from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from src.prompt import *
from werkzeug.utils import secure_filename
from pypdf import PdfReader


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY=os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY



embeddings = download_embeddings()
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings, 
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
def safe_retriever(query):
    results = retriever.invoke(query)
    fixed = []
    for r in results:
        if isinstance(r, dict):
            fixed.append(Document(page_content=r.get("text", str(r))))
        else:
            fixed.append(r)
    return fixed

ChatModel = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}")
    ]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": RunnableLambda(safe_retriever) | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatModel
    | StrOutputParser()
)


# --- upload folder ---
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)





@app.route('/get', methods=["GET", "POST"])
def chat():
    msg = request.form.get('msg', '')  
    if not msg:
        return "No message received", 400
    print(f"User message: {msg}")
    response = rag_chain.invoke(msg)
    if isinstance(response, dict):
        return str(response.get("answer", response))
    return str(response)


@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    
    # extract text based on file type
    extracted_text = ""
    try:
        if filename.lower().endswith('.pdf'):
            with open(file_path, 'rb') as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    extracted_text += page.extract_text() + "\n"
        elif filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg') or filename.lower().endswith('.png'):
            extracted_text = f"Medical image report: {filename}"
        else:
            extracted_text = f"Medical document: {filename}"
    except Exception as e:
        extracted_text = f"Could not read file content: {filename}"
    
    user_message = request.form.get('msg', '')
    prompt_text = f"Analyze this medical report:\n\n{extracted_text[:2000]}"
    if user_message:
        prompt_text += f"\n\nThe user also asks: {user_message}"
    
    response = rag_chain.invoke(prompt_text)
    return str(response)
        
        
        
        
        
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)