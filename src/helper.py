from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document


#Extract Data From the PDF File
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents



from typing import List
from langchain_core.documents import Document

def filter_tp_minimal_docs(docs: List[Document]) -> List[Document]:
    '''
    Given a list of documents objects, return a new list of documents 
    objects containing only the page_content and the source.
    '''
    minimal_docs: List[Document] = []
    
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
        
    return minimal_docs



# split the documents into smaller chunks
def text_splitter(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20, length_function=len)
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk



from langchain_huggingface import HuggingFaceEmbeddings

def download_embeddings():
    ''' 
    Download and return the HuggingFace embeddings model.
    '''
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
