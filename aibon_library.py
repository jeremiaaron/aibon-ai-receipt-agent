import os
import re
from dotenv import load_dotenv
load_dotenv()

import json
from logging import Logger
from psycopg import Connection
import traceback
from datetime import datetime
from pymongo import MongoClient

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

from prompts import extraction_prompt

class DocAgent():
    def __init__(self):
        pass
    
    def parse_with_gemini(self, file_name: str, image_bytes, logger: Logger, model_name: str="gemini-2.5-flash") -> dict:
        """
        Parse image information with Gemini model.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment.")

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=api_key,
            request_timeout=120
        )

        messages = []
        messages.append(SystemMessage(content=extraction_prompt))
        
        mime = "image/png" if file_name.endswith(".png") else "image/jpeg"
        messages.append(
            HumanMessage(
                content=[
                    {"type": "text", "text": "Extract this image"},
                    {"type": "media", "mime_type": mime, "data": image_bytes},
                ]
            )
        )
        resp = llm.invoke(messages)
        logger.debug(f"Raw extraction response: {resp}")
        
        if hasattr(resp, "content") and isinstance(resp.content, str):
            resp_str = resp.content.strip()
            resp_str = resp_str.replace("```json", "")
            resp_str = resp_str.replace("```", "")
            return json.loads(resp_str)
        
        resp_str = str(resp).strip()
        resp_str = resp_str.replace("```json", "")
        resp_str = resp_str.replace("```", "")
        return json.loads(resp_str)

class PostgresAgent():
    def __init__(self):
        pass
    
    def insert_data_to_db(self, conn: Connection, data: dict, logger: Logger):
        """
        Insert receipt data to Postgres DB.
        """
        try:
            with conn.cursor() as cursor:
                insert_receipts_query = """
                    INSERT INTO receipts (
                        merchant_name,
                        merchant_address,
                        merchant_phone,
                        transaction_date,
                        transaction_time,
                        receipt_number,
                        subtotal,
                        tax_and_fees,
                        discount,
                        grand_total
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                """
                
                merchant_info = data["merchant_details"]
                transaction_info = data["transaction_details"]
                items_info = data["line_items"]
                summary_info = data["cost_summary"]
                
                cursor.execute(
                    query=insert_receipts_query,
                    params=(
                        merchant_info["name"],
                        merchant_info["address"],
                        merchant_info["phone_number"],
                        transaction_info["date"],
                        transaction_info["time"],
                        transaction_info["receipt_number"],
                        summary_info["subtotal"],
                        summary_info["total_taxes_and_fees"],
                        summary_info["total_discount"],
                        summary_info["grand_total"]
                    )
                )
                
                receipt_id = cursor.fetchone()[0]
                
                insert_receipt_items_query = """
                    INSERT INTO receipt_items (
                        receipt_id,
                        item_name,
                        quantity,
                        unit_price,
                        total_price
                    )
                    VALUES (%s, %s, %s, %s, %s)
                """
                
                params = [
                    (
                        receipt_id,
                        item["item_name"],
                        item["quantity"],
                        item["unit_price"],
                        item["total_price"]
                    )
                    for item in items_info
                ]
                
                cursor.executemany(
                    query=insert_receipt_items_query,
                    params_seq=params
                )

            conn.commit()
            return receipt_id
        
        except Exception as e:
            logger.error("Error occurred when inserting data to DB.")
            logger.error(traceback.format_exc())
            return None

        
class MongoAgent():
    def __init__(self):
        mongo_user = os.getenv("MONGO_USER")
        mongo_pass = os.getenv("MONGO_PASS")
        self.client = MongoClient(f"mongodb://{mongo_user}:{mongo_pass}@mongodb:27017/")
        self.db = self.client["aibondb"]
        self.chat_collection = self.db["chat_history"]
        self.prompts_collection = self.db["prompts"]
        
    def get_all_chat_session_history(self, user_id: str):
        """
        Retrieve all chat session history (up to 15).
        """
        session_history = list(self.chat_collection.find({
            "user_id": user_id
        }).sort("last_modified", -1).limit(15))
        return session_history
    
    def get_one_chat_session_history(
        self,
        user_id: str,
        session_id: str
    ):
        """
        Retrieve only one chat session history.
        """
        chat_history = self.chat_collection.find_one({"user_id": user_id, "session_id": session_id})
        return chat_history
    
    def create_chat_session(
        self,
        user_id: str,
        session_id: str
    ) -> None:
        """
        Create a new chat session.
        """
        now = datetime.now()
        chat_data = {
            "_id": session_id,
            "user_id": user_id,
            "session_id": session_id,
            "date_created": now,
            "last_modified": now,
            "messages": []
        }
        self.chat_collection.insert_one(chat_data)
        
    def check_if_chat_session_exists(
        self,
        user_id: str,
        session_id: str
    ) -> bool:
        """
        Check if a chat session already exists or not.
        """
        sessions = list(self.chat_collection.find(
            {
                "user_id": user_id,
                "session_id": session_id
            },
            {
                "timestamp": 1,
                "messages": 1
            }
        ))
        
        if sessions:
            session = sessions[0]
            if "messages" in session and len(session["messages"]) > 0:
                return True
        return False
        
    def update_human_message(
        self,
        session_id: str,
        message_id: str,
        message: str
    ) -> None:
        """
        Update human message in an existing chat session.
        """
        now = datetime.now()
        self.chat_collection.update_one(
            {
                '_id': session_id,
            },
            {
                "$set": {
                    'last_modified': now
                },
                "$push": {'messages': {
                    'message_id': message_id,
                    'human_timestamp': now,
                    'human_message': message,
                    'ai_timestamp': None,
                    'ai_message': "",
                    'ai_message_feedback': "neutral",
                    'agents_executed': [],
                    'metadata': {}
                }}
            },
        )
         
    def update_ai_message(
        self,
        session_id: str,
        message_id: str,
        message: str
    ) -> None:
        """
        Update AI message in an existing chat session.
        """
        now = datetime.now()
        self.chat_collection.update_one(
            {
                '_id': session_id,
                'messages.message_id': message_id,
            },
            {
                "$set": {
                    'last_modified': now,
                    "messages.$.ai_timestamp": now,
                    "messages.$.ai_message": message
                }
            },
        )
        
    def update_message_metadata(
        self,
        session_id: str,
        message_id: str,
        metadata: dict
    ):
        """
        Update the metadata for a message (token count, model info, and other info).
        """
        self.chat_collection.update_one(
            {
                '_id': session_id,
                'messages.message_id': message_id,
            },
            {
                "$set": {"messages.$.metadata": metadata}
            },
        )
        
    def _tokenize(self, text):
        """
        Simple tokenizer that lowercases text, removes punctuation, and splits by whitespace.
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        return tokens

    def _partial_word_match_score(self, query, text):
        """
        Computes a matching score based on for search chat history function.
        """
        query_tokens = self._tokenize(query)
        text_tokens = self._tokenize(text)
        score = 0

        for q in query_tokens:
            for t in text_tokens:
                if q in t:
                    score += 1
                    break
        return score    
    
    # Search chat history function, not implemented yet in the UI
    def search_chat_history(
        self,
        query: str,
        top_k: int=0
    ):
        """
        Searches chat histories using simple word matching.
        Returns the top_k chat messages based on the number of matching words.
        """
        results = []

        for session in self.chat_collection.find({"user_id": self.user_id}):
            session_id = session.get('session_id')
            messages = session.get('messages', [])
            for message in messages:
                human_text = message.get('human_message', '')
                ai_text = message.get('ai_message', '')
                combined_text = f"{human_text} {ai_text}"
                
                score = self._partial_word_match_score(query, combined_text)
                if score > 0:
                    results.append({
                        'session_id': session_id,
                        'message': message,
                        'score': score
                    })
        
        if top_k > 0:
            return results[:top_k]
        else:
            return results
