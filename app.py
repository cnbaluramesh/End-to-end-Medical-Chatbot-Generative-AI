from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

logger.info("Starting Medical Chatbot application")

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()
logger.info("Embeddings downloaded successfully")


index_name = "medical-bot"
logger.info(f"Loading index: {index_name}")

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
logger.info("Pinecone vector store loaded successfully")

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
logger.info("Retriever configured successfully")


llm = ChatOpenAI(temperature=0.4, max_tokens=500)
logger.info("ChatOpenAI LLM initialized")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
logger.info("RAG chain created successfully")


@app.route("/")
def index():
    logger.info("Rendering chat.html")
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    logger.info(f"Received question: {msg}")

    try:
        response = rag_chain.invoke({"input": msg})
        answer = response["answer"]
        logger.info(f"Generated response: {answer[:100]}...")
        return str(answer)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return "Sorry, I encountered an error processing your request. Please try again."




if __name__ == '__main__':
    logger.info("Starting Flask server on 0.0.0.0:8080")
    app.run(host="0.0.0.0", port= 8080, debug= True)
