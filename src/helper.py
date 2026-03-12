from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import logging

# Configure logging
logger = logging.getLogger(__name__)


#Extract Data From the PDF File
def load_pdf_file(data):
    logger.info(f"Loading PDF files from directory: {data}")
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()
    logger.info(f"Loaded {len(documents)} documents from PDF files")
    return documents



#Split the Data into Text Chunks
def text_split(extracted_data):
    logger.info(f"Splitting {len(extracted_data)} documents into text chunks")
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    logger.info(f"Created {len(text_chunks)} text chunks")
    return text_chunks



#Download the Embeddings from HuggingFace
def download_hugging_face_embeddings():
    logger.info("Downloading HuggingFace embeddings model")
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
    logger.info("HuggingFace embeddings model loaded successfully")
    return embeddings