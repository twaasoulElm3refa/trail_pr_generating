from fastapi import FastAPI
#import mysql.connector
from connection_test import check_mysql_connection,fetch_press_releases
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

load_dotenv()

host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")


# ✅ 1. إعداد CORS Middleware (اختياري لكن مهم لو عندك Frontend خارجي)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ← أو ضع دومينات محددة مثل ["https://yourfrontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 2. Middleware لتسجيل كل طلب ورد
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"📥 Request: {request.method} {request.url}")
    response = await call_next(request)
    print(f"📤 Response status: {response.status_code}")
    return response


@app.get("/{user_id}")
async def root(user_id: str):
    connection =check_mysql_connection()
    #cursor = connection.cursor(dictionary=True)
    if connection is None:
        print("Failed to establish database connection")  # لن تظهر إذا الاتصال ناجح
    else:
        user_session_id = user_id
        all_release = fetch_press_releases(user_session_id)
        if not all_release:
            return {"error": "لا توجد نتائج في all_release"}
        release = all_release[-1]
    
        connection.commit()
        connection.close()

    return {"All_release":**all_release, "last release": release}

if __name__ == "__main__":              
    uvicorn.run(app, host=host, port=port)

