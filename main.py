from fastapi import FastAPI
#import mysql.connector
from connection_test import check_mysql_connection
import os
from dotenv import load_dotenv
import uvicorn

app = FastAPI()

load_dotenv()

host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")


@app.get("/")
async def generate_article(user_id: str):
    connection = get_db_connection()
    '''cursor = connection.cursor(dictionary=True)
    user_session_id = user_id
    all_release = fetch_press_releases(user_session_id)
    release = all_release[-1]

    # Prepare the Arabic prompt
    print(release['about_press'])

    connection.commit()
    cursor.close()
    connection.close()'''

    return {"connection":connection}

if __name__ == "__main__":              
    uvicorn.run(app, host=host, port=port)
