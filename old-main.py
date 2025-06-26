'''from fastapi import FastAPI
import openai
import numpy as np
import json
import faiss

app = FastAPI()

def generate_article(topic, lines_number, website):
    # تحميل داخل الدالة فقط لتقليل استهلاك الذاكرة عند التشغيل
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

    with open('filtered_corpus.json', 'r', encoding='utf-8') as json_file:
        corpus = json.load(json_file)

    index = faiss.read_index("my_index.index")
    
    topic_embedding = model.encode([topic])
    D, I = index.search(np.array(topic_embedding), 3)
    context = "\n".join([corpus[i]["content"] for i in I[0]])

    prompt =  f"""
أنت صحفي عربى محترف في مؤسسة إعلامية بارزة، ومتخصص في كتابة البيانات الإخبارية بلغة عربية فصيحة مع الالتزام بالبيانات والتفاصيل الممنوحة اليك وصياغتها فى صوره بيان مع الالتزام بعدد الاسطر 7 اسطر معتمدا فى البيان على البيانات المدخله لك من المستخدم او مواقع رسميه ذات مصادر موثقه مائة باللمائة مع ذكر فى بدايه البيان العنوان الرئيسي و تاريخ اليوم حسب الوطن العربي ثم محتوى البيان ثم كلمة 'للمحررين'  ثم الحول ثم فى نهايه البيان بيانات التواصل من تليفون و ايميل و وسائل التواصل الاجتماعى (ان ذكرت فى التلقين او البيانات المدخله من المستخدم) دون تاليف او تعديل مع ترك مساحتها فارغه اذا لم يتم تحديدها من المستخدم: {topic}.
    استخدم المعلومات التالية كنموذج لكيقية صياغه المبيان :
    {context}
    و الرجوع الى موقعهم الموجود فى   لكتابه حول عنهم من خدمات يقدموها الى من هم 
    """
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

@app.get("/")
async def root():
    article = "hello"
    print (article)
    return {"article":article }

@app.get("/{used_id}")
async def root(user_id: str):
    # Prepare the Arabic prompt
    topic = f" اكتب بيان عن احداث الحرب الحالية"
    article = generate_article_based_on_topic(topic)
    
    print(article)
    return {"article":article }




from fastapi import FastAPI
#from sentence_transformers import SentenceTransformer
#from pydantic import BaseModel
import faiss
import numpy as np
import openai
import json
import os

app = FastAPI()

#model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
#model = SentenceTransformer("paraphrase-multilingual-MiniLM-L6-v2")
#model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
#model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")  # الأصغر حجماً
model = SentenceTransformer("my_model")
with open('filtered_corpus.json', 'r', encoding='utf-8') as json_file:
    corpus = json.load(json_file)
index = faiss.read_index("my_index.index")

def get_openai_embedding(text: str):
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    return np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
    
def generate_article(topic):
    topic_embedding = = get_openai_embedding(topic)
    D, I = index.search(np.array(topic_embedding), 3)
    context = "\n".join([corpus[i]["content"] for i in I[0]])

    prompt = f"""
أنت صحفي عربى محترف في مؤسسة إعلامية بارزة، ومتخصص في كتابة البيانات الإخبارية بلغة عربية فصيحة مع الالتزام بالبيانات والتفاصيل الممنوحة اليك وصياغتها فى صوره بيان مع الالتزام بعدد الاسطر 7 اسطر معتمدا فى البيان على البيانات المدخله لك من المستخدم او مواقع رسميه ذات مصادر موثقه مائة باللمائة مع ذكر فى بدايه البيان العنوان الرئيسي و تاريخ اليوم حسب الوطن العربي ثم محتوى البيان ثم كلمة 'للمحررين'  ثم الحول ثم فى نهايه البيان بيانات التواصل من تليفون و ايميل و وسائل التواصل الاجتماعى (ان ذكرت فى التلقين او البيانات المدخله من المستخدم) دون تاليف او تعديل مع ترك مساحتها فارغه اذا لم يتم تحديدها من المستخدم: {topic}.
    استخدم المعلومات التالية كنموذج لكيقية صياغه المبيان :
    {context}
    و الرجوع الى موقعهم الموجود فى   لكتابه حول عنهم من خدمات يقدموها الى من هم 
    """
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

@app.get("/")
async def root():
    article = "hello"
    print (article)
    return {"article":article }

@app.get("/{used_id}")
async def root(user_id: str):
    # Prepare the Arabic prompt
    topic = f" اكتب بيان عن احداث الحرب الحالية"
    article = generate_article(topic)
    #article = generate_article_based_on_topic(topic)
    
    print(article)
    return {"article":article }
# return {"result": generate_article(data["topic"], data["lines_number"], data["website"])}

'''

'''from fastapi import FastAPI
#import mysql.connector
from connection_test import check_mysql_connection,fetch_press_releases ,update_press_release
from dotenv import load_dotenv
import os
import uvicorn
import asyncio
from pydantic import BaseModel
import openai
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

app = FastAPI()

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")

def generate_article_based_on_topic(topic, corpus, index):

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    # Find relevant documents based on the topic embedding
    topic_embedding = model.encode([topic])
    D, I = index.search(np.array(topic_embedding), 3)  # Get top 3 related documents

    # Retrieve the content for those documents
    context = "\n".join([corpus[i]["content"] for i in I[0]])
    

    # Create the prompt for GPT
    prompt = f"""
أنت صحفي عربى محترف في مؤسسة إعلامية بارزة، ومتخصص في كتابة البيانات الإخبارية بلغة عربية فصيحة مع الالتزام بالبيانات والتفاصيل الممنوحة اليك وصياغتها فى صوره بيان مع الالتزام بعدد الاسطر 7 اسطر معتمدا فى البيان على البيانات المدخله لك من المستخدم او مواقع رسميه ذات مصادر موثقه مائة باللمائة مع ذكر فى بدايه البيان العنوان الرئيسي و تاريخ اليوم حسب الوطن العربي ثم محتوى البيان ثم كلمة 'للمحررين'  ثم الحول ثم فى نهايه البيان بيانات التواصل من تليفون و ايميل و وسائل التواصل الاجتماعى (ان ذكرت فى التلقين او البيانات المدخله من المستخدم) دون تاليف او تعديل مع ترك مساحتها فارغه اذا لم يتم تحديدها من المستخدم: {topic}.
    استخدم المعلومات التالية كنموذج لكيقية صياغه المبيان :
    {context}
    و الرجوع الى موقعهم الموجود فى   لكتابه حول عنهم من خدمات يقدموها الى من هم 
    """

    # Get response from OpenAI
    response  = openai.chat.completions.create(
      model="gpt-4o-mini",
      store=True,
      messages=[{"role": "user", "content": prompt}]
        )
    
    return response.choices[0].message.content.strip()

@app.get("/{user_id}")
async def root(user_id: str):
    with open('filtered_corpus.json', 'r', encoding='utf-8') as json_file:
        corpus = json.load(json_file)
    index = faiss.read_index("my_index.index")
    
    # Prepare the Arabic prompt
    topic = f" اكتب بيان عن احداث الحرب الحالية"
    article = generate_article_based_on_topic(topic, corpus, index)
    
    print(article)
    
    return {"article":article }'''

if __name__ == "__main__":              
    uvicorn.run(app, host=host, port=port)
