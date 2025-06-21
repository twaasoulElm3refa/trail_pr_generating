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
            port=int(os.getenv("DB_PORT", 3306))
        )

        if connection.is_connected():
            print("✅ Connected!")
        else:
            print("❌ Failed.")

    except Error as e:
        print("❌ Error:", e)

check_mysql_connection()
