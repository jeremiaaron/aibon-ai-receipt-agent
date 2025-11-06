from psycopg import Connection
import traceback
import os
from dotenv import load_dotenv
load_dotenv()
import psycopg
from psycopg.rows import dict_row

def retrieve_table_schema(conn: Connection, table_name: str):
    print("Retrieving table schema")
    try:
        with conn.cursor() as cursor:
            schema_query = f"""
                SELECT 
                    table_name,
                    column_name,
                    data_type,
                    is_nullable,
                    column_default
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                ORDER BY table_name, ordinal_position;
            """
            cursor.execute(
                query=schema_query
            )
            result = cursor.fetchall()
        conn.commit()
        return result
    
    except Exception as e:
        print(traceback.format_exc())
        
def main():
    conn: Connection = psycopg.connect(
        host="localhost",
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        row_factory=dict_row
    )
    
    receipts_schema = retrieve_table_schema(
        conn=conn,
        table_name="receipts"
    )
    
    print(receipts_schema)
    
    # just for GitHub actions testing
    
if __name__ == "__main__":
    main()