import os
from dotenv import load_dotenv
load_dotenv()

import random
import time
from datetime import datetime
import json
from typing import (
    Annotated,
    Sequence,
    TypedDict,
    Iterator,
    Any
)
from logging import Logger
import uuid
import pytz

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import (
    BaseMessage,
    HumanMessage
)
from langchain_core.runnables import RunnableConfig, Runnable

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.config import get_stream_writer

from langgraph.types import Command

from aibon_library import MongoAgent
from prompts import generate_sql_query_prompt, aibon_system_prompt

from openai import RateLimitError

from psycopg import Connection

parser = StrOutputParser()

def load_llm(model, temperature=0.5, max_output_tokens=8192):
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        max_output_tokens=max_output_tokens,
    )
    return llm

def invoke_chain(
    runnable_chain: Runnable,
    vars_input: dict | list[BaseMessage] = {},
    max_retries: int=3
):
    """
    Invoke LLM LangChain chain call with retries.
    """
    for _ in range(max_retries):
        try:
            response = runnable_chain.invoke(vars_input)
            return response
        except RateLimitError as e:
            if e.status_code == 429:
                wait = float(e.response.headers.get("Retry-After", 1))
                time.sleep(wait + random.random() * 0.5)
                continue
            raise
    raise RuntimeError(f"Exceeded maximum retry attempts ({max_retries})!")

def retrieve_table_schema(conn: Connection):
    """
    Retrieve 'receipts' and 'receipt_items' table schema from Postgres DB.
    """
    with conn.cursor() as cursor:
        schema_query = f"""
            SELECT 
                table_name,
                column_name,
                data_type,
                is_nullable,
                column_default
            FROM information_schema.columns
            WHERE table_name = 'receipts' OR table_name = 'receipt_items'
            ORDER BY table_name, ordinal_position;
        """
        cursor.execute(
            query=schema_query
        )
        result = cursor.fetchall()
    conn.commit()
    return result

def retrieve_sample_data(conn: Connection, table_name: str):
    """
    Retrieve sample data from a table in Postgres DB.
    """
    with conn.cursor() as cursor:
        schema_query = f"""
            SELECT 
                *
            FROM {table_name}
            LIMIT 1;
        """
        cursor.execute(
            query=schema_query
        )
        result = cursor.fetchall()
    conn.commit()
    return result

def run_sql_query(conn: Connection, sql_query: str):
    """
    Execute SQL query
    """
    with conn.cursor() as cursor:
        cursor.execute(
            query=sql_query
        )
        result = cursor.fetchall()
    conn.commit()
    return result


class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    

class AibonAgent():
    def __init__(
        self,
        user_id: str,
        session_id: str,
        message_id: str,
        sql_conn: Connection,
        logger: Logger,
    ):
        """
        Initialize variables for AibonAgent.
        """
        self.user_id = user_id
        self.session_id = session_id
        self.message_id = message_id
        self.sql_conn = sql_conn
        self.logger = logger
        
        self.llm = load_llm(model="gemini-2.5-flash", temperature=0.5, max_output_tokens=2048)
        self.mongo_agent = MongoAgent()
        
        self.language = "Indonesian"
        self.tz_region = "Asia/Jakarta"
        
        self.sql_query_result = "[]"
        
        self.agent = self.create_graph()
    
    def create_graph(self) -> CompiledStateGraph:
        """
        Create a graph of agent workflow using LangGraph's StateGraph.
        
        :return: the compiled state graph.
        """
        
        def generate_sql_query(state: AgentState):
            """
            A function to generate SQL query based on user request (except if the user's query is ambiguous or unrelated to the receipts database).
            
            :return: command which next node to go to and updates the agent state messages.
            """
            writer = get_stream_writer()
            writer({"agent_name": "aibon_agent", "activity_status": "generate_sql_query"})
            
            all_messages = state["messages"]
            filtered_messages = [msg for msg in all_messages if not(isinstance(msg, HumanMessage) and "is_ai_result" in msg.response_metadata)]
            
            table_schema = retrieve_table_schema(conn=self.sql_conn)
            sample_data_receipts = retrieve_sample_data(conn=self.sql_conn, table_name='receipts')
            sample_data_receipt_items = retrieve_sample_data(conn=self.sql_conn, table_name='receipt_items')

            messages_to_invoke = [SystemMessagePromptTemplate.from_template(generate_sql_query_prompt)] + filtered_messages
            chat_prompt = ChatPromptTemplate(
                messages=messages_to_invoke
            )
            llm = load_llm(model="gemini-2.5-flash", temperature=0.3, max_output_tokens=2048)
            chain = chat_prompt | llm | parser
            
            response = invoke_chain(
                runnable_chain=chain,
                vars_input={
                    "table_schema": table_schema,
                    "sample_data_receipts": sample_data_receipts,
                    "sample_data_receipt_items": sample_data_receipt_items
                },
                max_retries=3
            )
            self.logger.info(f"Generate SQL query result: {response}")
            response = response.replace("```json", "")
            response = response.replace("```", "")
            response_eval = json.loads(response)
            
            self.logger.info(f"Generate SQL query result: {response_eval}")
            writer({"agent_name": "aibon_agent", "activity_status": "generate_sql_query", "result": str(response_eval)})
            
            self.language = response_eval["language"]
            
            outputs = []
            if response_eval["action"] == "sql_query":
                outputs.append(
                    HumanMessage(
                        content=response_eval["sql_query"],
                        response_metadata={"is_ai_result": True, "function": "generate_sql_query"},
                        id=str(uuid.uuid4())
                    )
                )

                return Command(
                    update={
                        "messages": outputs
                    },
                    goto="execute_sql_query"
                )
            
            elif response_eval["action"] == "cancel":
                outputs.append(
                    HumanMessage(
                        content=f"The information cannot be extracted. Reason: {response_eval['reason']}",
                        response_metadata={"is_ai_result": True, "function": "generate_sql_query"},
                        id=str(uuid.uuid4())
                    )
                )
                return Command(
                    update={
                        "messages": [
                            HumanMessage(
                                content=f"The information cannot be extracted. Reason: {response_eval['reason']}",
                                response_metadata={"is_ai_result": True, "function": "generate_sql_query"},
                                id=str(uuid.uuid4())
                            )
                        ]
                    },
                    goto="formulate_response"
                )
                
            elif response_eval["action"] == "nothing":
                return Command(goto="formulate_response")
        
        def execute_sql_query(state: AgentState):
            """
            A function to execute the sql query generated by the LLM.
            
            :return: response output of the LLM.
            """
            writer = get_stream_writer()
            writer({"agent_name": "aibon_agent", "activity_status": "execute_sql_query"})
            
            last_message = state["messages"][-1]
            sql_query = last_message.content
            
            result = run_sql_query(
                conn=self.sql_conn,
                sql_query=sql_query
            )
            
            writer({"agent_name": "aibon_agent", "activity_status": "execute_sql_query", "execution_result": result})
            
            outputs = []
            outputs.append(
                HumanMessage(
                    content=f"SQL query result: {result}",
                    response_metadata={"is_sql_query_result": True, "function": "execute_sql_query"},
                    id=str(uuid.uuid4())
                )
            )
            
            return {"messages": outputs}
            
        def formulate_response(state: AgentState, config: RunnableConfig) -> AgentState:
            """
            A function to call the LLM based on the current agent state and the given system prompt.
            
            :return: response output of the LLM.
            """
            writer = get_stream_writer()
            writer({"agent_name": "jarvis_agent", "activity_status": "formulate_response"})

            tz_info = pytz.timezone(self.tz_region)
            datetime_now = datetime.now(tz_info)
            date_str = datetime_now.strftime("%Y-%m-%d %H:%M:%S")
            language = f"Reply in {self.language}"
            
            messages_to_invoke = [SystemMessagePromptTemplate.from_template(aibon_system_prompt)] + state["messages"]

            chat_prompt = ChatPromptTemplate(
                messages=messages_to_invoke
            )
            chain = chat_prompt | self.llm | parser
            
            response = invoke_chain(
                runnable_chain=chain,
                vars_input={
                    "datetime_now": date_str,
                    "tz_region": self.tz_region,
                    "language": language
                },
                max_retries=3
            )
            
            return {"messages": [response]}
    
        workflow = StateGraph(AgentState)

        workflow.add_node("generate_sql_query", generate_sql_query)
        workflow.add_node("execute_sql_query", execute_sql_query)
        workflow.add_node("formulate_response", formulate_response)

        workflow.set_entry_point("generate_sql_query")
        workflow.add_edge("execute_sql_query", "formulate_response")
        workflow.add_edge("formulate_response", END)

        graph = workflow.compile()
        
        return graph
    
    def generate_response(
        self, 
        inputs: AgentState, 
        stream: bool = True
    ) -> Iterator[dict[str, Any] | Any]:
        """
        Generate response based on the compiled agent flow.
        
        :param inputs: user messages as input for the agent.
        :param stream: enable text streaming or not.
        :return: a generator containing agent stream output.
        """
        if stream:
            result = self.agent.stream(
                inputs,
                stream_mode=["custom", "messages"]
            )
        else:
            result = self.agent.invoke(
                inputs
            )
        return result