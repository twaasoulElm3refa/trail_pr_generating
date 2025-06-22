import os
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error

load_dotenv()  # Load from .env

def check_mysql_connection():
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            port=int(os.getenv("DB_PORT"))
        )

        if connection.is_connected():
            print("✅ Connected!")
        else:
            print("❌ Failed.")

    except Error as e:
        print("❌ Error:", e)
        
def fetch_press_releases(user_id: str ):
    connection = get_db_connection()
    if connection is None:
        print("Failed to establish database connection")
        return []
    
    try:
        cursor = connection.cursor(dictionary=True)
        query = """
        SELECT * 
        FROM wpl3_press_release_Form
        WHERE user_id = %s 
        """
        cursor.execute(query, (user_id,))

        # Fetch the first row
        all_user_articles = cursor.fetchall()

        return all_user_articles

    except Error as e:
        print(f"Error fetching data: {e}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
#check_mysql_connection()
