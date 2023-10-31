import functools
import mimetypes
import pickle
import sys
from flask import Flask, json, request
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from hmac import compare_digest
from http.client import HTTPException
import sys
from flask import request
import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.llms import OpenAI
import uuid
import mimetypes
import pickle

from PyPDF2 import PdfReader
from ApiResponse import ApiResponse
from ApiResponse import ApiResponse
from models import DeviceModel
from utils import is_access_token_valid



app = Flask(__name__)
app.config.from_object('config')




def isValid(device_key):
    device = DeviceModel.find_by_device_key(device_key)
    if device and compare_digest(device.device_key, device_key):
        return True
    


def appIdRequired(func):
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        app.logger.debug(f"Is request a json ? {request.is_json}")
        app.logger.debug(f"The form data is {request.form}")
        app.logger.debug(f"The app_id is {request.form.get('app_id')}")
        if request.is_json:
            app_id = request.json.get("app_id")
        elif request.form:
            app_id = request.form.get("app_id")
        else:
            return ApiResponse.badRequest(
                app=app,
                content={
                    "app_id": "Please provide a valid app_id"
                }
            )
        # Check if app_id is correct and valid
        if request.method == "POST" and isValid(app_id):
            return func(*args, **kwargs)
        else:
            return ApiResponse.genericResponse(
                app=app,
                content={
                    "message": "Please provide a valid app_id"
                },
                code=403
            )
    return decorator


def accessTokenRequired(func):
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        try:
            token = request.headers.get("authorization").split(" ")[1].strip()
        except:
            return ApiResponse.genericResponse(
                app=app,
                content={
                    "message": "Access token should be provided as Bearer"
                },
                code = 401
            )

        # Check if the access_token is correct and valid
        if (is_access_token_valid(token)):
            return func(*args, **kwargs)
        else:
            return ApiResponse.genericResponse(
                app=app,
                content={
                    "message": f"The provided token: {token} is invalid"
                },
                code=403
            )
    return decorator

def processText(text):
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
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base


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


def makeDirs(path):
    if (not os.path.exists(path)):
        os.makedirs(path)


def saveKnowledgeBaseFrom(savedPath, application_id):
    pdf_reader = PdfReader(savedPath)
    # Text variable will store the pdf text
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Create the knowledge base object
    knowledgeBase = processText(text)
    knowledge_key = str(uuid.uuid4())
    base_path = os.path.join(os.environ.get(
        "SERIALIZED_OBJECT_FOLDER"), application_id)
    object_path = os.path.join(base_path, f"{knowledge_key}")
    makeDirs(base_path)
    with open(object_path, "wb") as file:
        pickle.dump(knowledgeBase, file)


def getKnowledgeBaseFrom(application_id):
    base_path = os.path.join(os.environ.get(
        "SERIALIZED_OBJECT_FOLDER"), application_id)
    files = os.listdir(base_path)
    for index, file in enumerate(files):
        with open(os.path.join(base_path, file), "rb") as object_file:
            if (index == 0):
                knowledge_base = pickle.load(object_file)
            else:
                current_base = pickle.load(object_file)
                knowledge_base.merge_from(current_base)
    app.logger.debug(f"The knowledge base is {knowledge_base}")
    return knowledge_base



@app.route('/')
def home():
    return "Hello world"



@app.route('/knowledge/upload', methods=['POST'])
@appIdRequired
def uploadFile():
    app_id = request.form.get("app_id", None)
    # append the uploaded file to the knowledge database using the key
    if 'pdf_file' in request.files:
        file = request.files['pdf_file']
        knowledge_key = str(uuid.uuid4())
        extension = getRealExtension(file)
        if (extension != ".pdf"):
            app.logger.debug("The extesion is not a pdf")
            # throw api error
        saved_path = os.path.join(os.environ.get("UPLOADED_FILE_FOLDER"))
        file_path = os.path.join(saved_path,  f"{knowledge_key}.pdf")
        makeDirs(saved_path)
        file.save(file_path)
        saveKnowledgeBaseFrom(file_path, app_id)
        os.remove(file_path)
        return ApiResponse.success(
            app = app,
            content = "File uploaded sucessfully"
        )



@app.route('/knowledge/test', methods=['POST'])
@appIdRequired
def answerMyQuestion():
    request_data = request.get_json()
    app_id = request_data["app_id"]
    # Check that a the current app_id has a knowledge base registered
    base_path = os.path.join(
        os.environ.get("SERIALIZED_OBJECT_FOLDER"),
        app_id
    )
    if (not os.path.exists):
        return ApiResponse.badRequest(
            app=app,
            content={
                "app_id": f"AppId: {app_id} invalid or the current app does not upload any valid documents"
            }
        )

    # Check that the question is passed as argument
    try:
        question = request_data["question"]
    except:
        return ApiResponse.badRequest(
            app=app,
            content={
                "question": "No question json attribute passed"
            }
        )

    if (question is None or not bool(question.strip())):
        return ApiResponse.badRequest(
            app=app,
            content={
                "question": "The question is empty, blank or null"
            }
        )
    # If it is, get completion using openapi api
    knowledge_base = getKnowledgeBaseFrom(app_id)
    docs = knowledge_base.similarity_search(question)
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type='stuff')

    with get_openai_callback() as cost:
        response = chain.run(input_documents=docs, question=question)
        app.logger.debug(cost)
    # return the response
    return response



@app.route('/register/device/', methods=['POST'])
@accessTokenRequired
def registerDevice():

    if request.is_json:
        device_name = request.json.get("device_name")
    elif request.form:
        device_name = request.form.get("device_name")
    else:
        return ApiResponse.badRequest(
            app=app,
            content={
                "device_name": "The device name is required"
            }
        )
    if DeviceModel.find_by_name(device_name):
        return ApiResponse.badRequest(
            app=app,
            content={
                "device_name": f"A device with name '{device_name}' already exists."
            }
        )

    new_device = DeviceModel(
        device_name=device_name,
    )
    new_device.save_to_db()

    return ApiResponse.genericResponse(
        app=app,
        content={
            "app_id": new_device.device_key
        },
        code=201
    )



@app.errorhandler(404)
def pageNotFound(error):
    return ApiResponse.genericResponse(
        app=app,
        content={
            "message": "Url invalid",
            "error": error.description
        },
        code=404
    )


@app.errorhandler(400)
def badRequest(error):
    return ApiResponse.badRequest(
        app=app,
        content=error.descripton,
    )


@app.errorhandler(500)
def internalServerError(error):
    _, exc_value, _ = sys.exc_info()

    # if (isinstance(exc_value, OpenAIError)) :
    #     app.logger.error("Open AI error")
    #     app.logger.info(exc_value.json_body or exc_value._message)
    return ApiResponse.genericResponse(
        app=app,
        content={
            "message": "Internal server error",
            "error": error.description
        },
        code=500
    )
