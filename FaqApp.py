from flask import request, Flask
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import uuid
import mimetypes
import pickle


# Load environment variables
load_dotenv()

def process_text(text):
    # Split the text into chunks using langchain
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = OpenAIEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    
    return knowledgeBase


def getRealExtension(file):
    # Get the content type from the request or default to 'application/octet-stream'
    content_type = file.content_type or 'application/octet-stream'

    # Get the file extension from the original file name
    filename = file.filename
    extension = filename.rsplit('.', 1)[-1]

    # Use the MIME type database to get a more accurate extension
    mime_extension = mimetypes.guess_extension(content_type)
    # Compare the two extensions and return the more accurate one
    if mime_extension:
        return mime_extension
    return extension

def make_dirs(path):
    if (not os.path.exists(path)):
        os.makedirs(path)

def getAndSaveKnowledgeBaseFrom(savedPath, application_id):
    pdf_reader = PdfReader(savedPath)
    # Text variable will store the pdf text
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # Create the knowledge base object
    knowledgeBase = process_text(text)
    knowledge_key = str(uuid.uuid4())
    base_path = os.path.join(os.environ.get("SERIALIZED_OBJECT_FOLDER"), application_id)
    object_path = os.path.join(base_path, f"{knowledge_key}")
    make_dirs(base_path)
    with open(object_path, "wb") as file:
        pickle.dump(knowledgeBase, file)


app = Flask(__name__)


@app.route('/')
def home():
    return "Hello world"

@app.route('/upload/<app_id>', methods=['POST'])
def uploadFile(app_id):
    # assert that the app_id is a  non empty string
    # append the uploaded file to the knowledge database using the key
    if 'pdf_file' in request.files:
        file = request.files['pdf_file']
        knowledge_key = str(uuid.uuid4())
        extension = getRealExtension(file)
        if (extension != ".pdf"):
            print("The extesion is not a pdf")
            # throw api error
        saved_path = os.path.join(os.environ.get("UPLOADED_FILE_FOLDER"))
        file_path = os.path.join(saved_path,  f"{knowledge_key}.pdf")
        make_dirs(saved_path)
        file.save(file_path)
        getAndSaveKnowledgeBaseFrom(file_path, app_id)
        os.remove(file_path)
        return f"File uploaded: {file_path}"


@app.route('/answer/<app_id>', methods=['POST'])
def answerMyQuestion(app_id):
    request_data = request.get_json()
    question = request_data["question"]
    if (question == None):
        return
    # Check that a the current app_id has a knowledge base registered
    # If not, throw error
    # Else, check that the question is passed as argument
    # If it is, get completion using openapi api
    #return the response
    

if __name__ == '__main__':
    app.run(debug = True)