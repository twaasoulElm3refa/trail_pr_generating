from fastapi import FastAPI
#import mysql.connector
from connection_test import check_mysql_connection,fetch_press_releases
import os
from dotenv import load_dotenv
import uvicorn

app = FastAPI()

load_dotenv()

host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")


@app.get("/{user_id}")
async def root(user_id: str):
    connection =check_mysql_connection()
    cursor = connection.cursor(dictionary=True)
    user_session_id = user_id
    all_release = fetch_press_releases(user_session_id)
    release = all_release[-1]

    # Prepare the Arabic prompt
    print(release['about_press'])

    connection.commit()
    cursor.close()
    connection.close()

    return {"Press release":release['about_press']}

if __name__ == "__main__":              
    uvicorn.run(app, host=host, port=port)

'''@app.get("/")
async def root():
    connection = check_mysql_connection()
    cursor = connection.cursor(dictionary=True)
    user_session_id = user_id
    all_release = fetch_press_releases(user_session_id)
    release = all_release[-1]

    # Prepare the Arabic prompt
    print(release['about_press'])

    connection.commit()
    cursor.close()
    connection.close()

    return {"connection":"done"}

if __name__ == "__main__":              
    uvicorn.run(app, host=host, port=port)'''
