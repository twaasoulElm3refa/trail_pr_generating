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
            return connection
        else:
            print("❌ Failed.")
            return None

    except Error as e:
        print("❌ Error:", e)
        return None
        
def fetch_press_releases(user_id: str ):
    connection = check_mysql_connection()
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
        print(len(all_user_articles ))

        return all_user_articles

    except Error as e:
        print(f"Error fetching data: {e}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def update_press_release(user_id, organization_name, article):
    connection = check_mysql_connection()
    if connection is None:
        return False
    
    try:
        cursor = connection.cursor()
        query = """
        INSERT INTO wpl3_articles (user_id, organization_name, article)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE article = VALUES(article)
        """
        cursor.execute(query, (user_id, organization_name, article))
        connection.commit()
        return True
    except Error as e:
        print(f"Error updating data: {e}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

#check_mysql_connection()
