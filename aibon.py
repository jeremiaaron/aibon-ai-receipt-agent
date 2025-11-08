import streamlit as st
from datetime import datetime
import pytz
import os
from dotenv import load_dotenv
load_dotenv()

import logging
from logging.handlers import RotatingFileHandler
from logging import Logger
import traceback
import uuid
import time

import psycopg
from psycopg import Connection

from langchain.schema.messages import AIMessage, AIMessageChunk, HumanMessage

from aibon_library import DocAgent, MongoAgent, PostgresAgent
from ai_agent import AibonAgent

st.set_page_config(page_title="AI-Bon Agent", layout="wide")
st.title("AI-Bon, Your Friendly AI Receipt Assistant!")

# arbitrary user_id, only 1 user for prototype
# persist users data if the project is scaled for multiple users
user_id = "12345678"
mongo_agent = MongoAgent()

# logging configuration
os.makedirs("logs", exist_ok=True)

class TZFormatter(logging.Formatter):
    def converter(self, timestamp):
        dt = datetime.fromtimestamp(timestamp, pytz.utc)
        target_tz = pytz.timezone('Asia/Bangkok')
        return dt.astimezone(target_tz)

    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not logger.handlers:
    root_handler = RotatingFileHandler(
        'logs/aibon.log',
        maxBytes=10*1024*1024,
        backupCount=5
    )
    root_formatter = TZFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    root_handler.setFormatter(root_formatter)
    logger.addHandler(root_handler)

# streamlit session state initialization
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "session_id" not in st.session_state:
    st.session_state["session_id"] = None
if "selected_session" not in st.session_state:
    st.session_state["selected_session"] = None
if "pg_connection" not in st.session_state:
    logger.info("Loading PostgreSQL connection...")
    conn: Connection = psycopg.connect(
        host="postgresdb",
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD")
    )
    st.session_state["pg_connection"] = conn
    logger.info("Connection completed.")
    
def add_human_message_to_db(message: str, user_id: str, session_id: str, message_id: str) -> None:
    if not mongo_agent.check_if_chat_session_exists(
        user_id=user_id,
        session_id=session_id
    ):
        mongo_agent.create_chat_session(
            user_id=user_id,
            session_id=session_id
        )
    mongo_agent.update_human_message(
        session_id=session_id,
        message_id=message_id,
        message=message
    )
    
def add_ai_message_to_db(message: str, session_id: str, message_id: str) -> None:
    mongo_agent.update_ai_message(
        session_id=session_id,
        message_id=message_id,
        message=message
    )

def format_history_mongo_to_streamlit(raw_history: dict):
    formatted_messages = []
    messages = raw_history["messages"]
    
    for msg in messages:
        if "human_message" in msg:
            if msg["human_message"] == "":
                msg_new = "-"
            else:
                msg_new = msg["human_message"]
            formatted_messages.append({"role": "user", "content": msg_new, "message_id": msg["message_id"]})
        if "ai_message" in msg:
            if msg["ai_message"] == "":
                msg_new = "-"
            else:
                msg_new = msg["ai_message"]
            formatted_messages.append({"role": "assistant", "content": msg_new, "message_id": msg["message_id"]})
        
    return formatted_messages

def format_history_mongo_to_langchain(raw_history: list):
    history_list = []
    
    for msg in raw_history:
        if "human_message" in msg:
            if msg["human_message"] == "":
                msg_new = "-"
            else:
                msg_new = msg["human_message"]
            history_list.append(HumanMessage(msg_new))
        if "ai_message" in msg:
            interrupt_status = {}
            if msg["ai_message"] == "":
                msg_new = "-"
            else:
                msg_new = msg["ai_message"]
            history_list.append(AIMessage(msg_new, response_metadata={"interrupt_status": interrupt_status}))
    return history_list

def extract_data_from_receipt(file_name: str, image_bytes):
    try:
        doc_agent = DocAgent()
        extracted_data = doc_agent.parse_with_gemini(
            file_name=file_name,
            image_bytes=image_bytes,
            logger=logger
        )
        return extracted_data
    except Exception as e:
        logger.error("Error occurred when extracting data from receipt.")
        logger.error(traceback.format_exc())
        
def insert_data_to_db(data: dict):
    try:
        postgres_agent = PostgresAgent()
        conn: Connection = st.session_state["pg_connection"]
        inserted_id = postgres_agent.insert_data_to_db(
            conn=conn,
            data=data,
            logger=logger
        )
        return inserted_id
    except Exception as e:
        logger.error("Error occurred when inserting extracted receipt data to DB.")
        logger.error(traceback.format_exc())

with st.sidebar:
    if st.button("New Chat"):
        st.session_state["session_id"] = None
        st.session_state["selected_session"] = None
        st.session_state["messages"] = []
        
    model_choice = st.selectbox("AI Chat Model", ["Gemini 2.5 Flash", "OpenAI GPT4.1-mini"], index=0, disabled=True)
    if model_choice == "Gemini 2.5 Flash":
        chosen_model = "gemini-2.5-flash"
    else:
        chosen_model = "gpt-4.1-mini"
    
    uploaded_files = st.file_uploader("Upload your receipt here:", accept_multiple_files=True, type=["png", "jpg", "jpeg"])
    load_button = st.button("Load receipt data into DB")
    
    if load_button:
        with st.spinner(f"Loading {len(uploaded_files)} receipt data, just a moment..."):
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    image_bytes = uploaded_file.getvalue()
                    file_name = uploaded_file.name
                    st.info(f"Processing receipt data {i+1} with file {file_name}")
                    
                    logger.info("Extracting receipt data...")
                    extracted_data = extract_data_from_receipt(file_name=file_name, image_bytes=image_bytes)
                    logger.info(f"Extracted data: {extracted_data}")
                    
                    logger.info("Inserting receipt data to DB...")
                    inserted_id = insert_data_to_db(data=extracted_data)
                    
                    if not inserted_id:
                        logger.error("Insertion failed: inserted_id is null.")
                        st.error("Oops! An error occurred while loading your receipt data, please try again later.")
                    else:  
                        logger.info(f"Inserted receipt id: {inserted_id}")
                        st.success(f"Successfully extracted and loaded receipt data with ID {inserted_id}!")
                        
                except Exception as e:
                    logger.error("An unknown error occurred when loading receipt data to DB.")
                    logger.error(traceback.format_exc())
                    st.error("Oops! An error occurred while loading your receipt data, please try again later.")
                    
    st.markdown("**:blue[Session History]**")
    history = mongo_agent.get_all_chat_session_history(user_id=user_id)
    for i, hist in enumerate(history):
        session_button_text = ""
        session_messages = hist.get("messages", [])
        if session_messages:
            session_button_text = session_messages[0].get("human_message", "")
        if st.button(session_button_text, key=f"session_button_{i}"):
            st.session_state["selected_session"] = hist
            st.session_state["session_id"] = hist["_id"]

if st.session_state["selected_session"]:
    st.session_state["messages"] = format_history_mongo_to_streamlit(
        raw_history=st.session_state["selected_session"]
    )
    
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask AI-Bon about your receipts!")
if user_input:
    # update streamlit displayed messages state
    st.session_state["messages"].append({"role": "user", "content": user_input})
    
    new_session = False
    if not st.session_state["session_id"]:
        new_session = True
        st.session_state["session_id"] = str(uuid.uuid4())
        
    message_id = str(uuid.uuid4())
    
    add_human_message_to_db(
        message=user_input,
        user_id=user_id,
        session_id=st.session_state["session_id"],
        message_id=message_id
    )
    
    with st.chat_message("user"):
        st.markdown(user_input)
        
    aibon_agent = AibonAgent(
        user_id=user_id,
        session_id=st.session_state["session_id"],
        message_id=message_id,
        sql_conn=st.session_state["pg_connection"],
        logger=logger
    )
        
    complete_history = mongo_agent.get_one_chat_session_history(
        user_id=user_id,
        session_id=st.session_state["session_id"]
    )
    if complete_history and "messages" in complete_history:
        chat_history = complete_history["messages"]
    else:
        chat_history = []
    
    if len(chat_history) > 0:
        temp_history = format_history_mongo_to_langchain(raw_history=chat_history)
        messages_to_invoke = \
            temp_history + \
            [HumanMessage(user_input)]
    else:
        messages_to_invoke = [HumanMessage(user_input)]
    
    stream = aibon_agent.generate_response(
        inputs={"messages": messages_to_invoke}
    )
    
    final_response = ""
    total_input_tokens = 0
    total_output_tokens = 0
    
    start_time = time.time()
    with st.chat_message("assistant"):
        with st.spinner("AIBon is thinking..."):
            stream_placeholder = st.empty()
            
            for output_type, output in stream:
                if output_type == "custom":
                    logger.info(f"Currently running process: {output}")
                    if "token_usage" in output:
                        process = output["token_usage"]["process"]
                        total_input_tokens += output["token_usage"]["input_tokens"]
                        total_output_tokens += output["token_usage"]["output_tokens"]
                        # logger.info(f"Process: {process} | Input tokens: {output['token_usage']['input_tokens']} | Output tokens: {output['token_usage']['output_tokens']}")

                elif output_type == "messages":
                    message, metadata = output
                    # token tracking for gemini
                    if "gemini" in chosen_model and isinstance(message, AIMessageChunk) and "finish_reason" in message.response_metadata \
                        and message.response_metadata["finish_reason"] == "STOP" and message.usage_metadata:
                        total_input_tokens += message.usage_metadata["input_tokens"]
                        total_output_tokens += message.usage_metadata["output_tokens"]
                    
                    # token tracking for openai gpt
                    elif "gpt" in chosen_model and isinstance(message, AIMessageChunk) and message.usage_metadata:
                        total_input_tokens += message.usage_metadata["input_tokens"]
                        total_output_tokens += message.usage_metadata["output_tokens"]

                    if metadata["langgraph_node"] == "formulate_response":
                        if isinstance(message, AIMessageChunk): 
                            if "finish_reason" in message.response_metadata and message.response_metadata["finish_reason"].lower() == "stop":
                                completion_time = round(time.time() - start_time, 2)
                            final_response += message.content
                        else:
                            final_response += message
                            
                        stream_placeholder.markdown(final_response)

    # update streamlit displayed messages state
    st.session_state["messages"].append({"role": "assistant", "content": final_response})

    # update AI chat in chat history
    add_ai_message_to_db(
        message=final_response,
        session_id=st.session_state["session_id"],
        message_id=message_id
    )
    
    message_metadata = {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "completion_time": round(time.time() - start_time, 2),
        "model_name": chosen_model,
        "model_version": "0.1",
    }
    mongo_agent.update_message_metadata(
        session_id=st.session_state["session_id"],
        message_id=message_id,
        metadata=message_metadata
    )
    
    current_selected_session = st.session_state["selected_session"]
    st.session_state["selected_session"] = mongo_agent.get_one_chat_session_history(
        user_id=user_id,
        session_id=st.session_state["session_id"]
    )
    
    if new_session:
        st.rerun()