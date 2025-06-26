from fastapi import FastAPI
#import mysql.connector
from connection_test import check_mysql_connection,fetch_press_releases ,update_press_release
import os
from dotenv import load_dotenv
#from fastapi.middleware.cors import CORSMiddleware
import uvicorn
#import asyncio
#from pydantic import BaseModel
import openai
#import faiss
#import numpy as np
#import json
#from sentence_transformers import SentenceTransformer


app = FastAPI()

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")


def generate_article_based_on_topic(topic,context,lines_number,website):
   
    # Create the prompt for GPT
    prompt = f"""
أنت صحفي عربى محترف في مؤسسة إعلامية بارزة، ومتخصص في كتابة البيانات الإخبارية بلغة عربية فصيحة تقوم بانشاء بيان صحفي نيابة عنهم حيث تصيغ البيان بصيغة "تعلن شركة ..."وليس "اعلنت" مع الالتزام بالبيانات والتفاصيل الممنوحة اليك وصياغتها فى صوره بيان مع الالتزام بعدد الاسطر {lines_number} معتمدا فى البيان على البيانات المدخله لك من المستخدم او مواقع رسميه ذات مصادر موثقه مائة باللمائة مع ذكر فى بدايه البيان العنوان الرئيسي و تاريخ اليوم حسب الوطن العربي ثم محتوى البيان ثم كلمة 'للمحررين'  ثم الحول ثم فى نهايه البيان بيانات التواصل من تليفون و ايميل و وسائل التواصل الاجتماعى (ان ذكرت فى التلقين او البيانات المدخله من المستخدم) دون تاليف او تعديل مع ترك مساحتها فارغه اذا لم يتم تحديدها من المستخدم: {topic}.
    استخدم المعلومات التالية كنموذج لكيقية صياغه المبيان :
    {context}
    و الرجوع الى موقعهم الموجود فى  {website} لكتابه حول عنهم من خدمات يقدموها الى من هم 
    """  
   
    # Get response from OpenAI
    response  = openai.chat.completions.create(model="gpt-4o-mini",
                                               store=True,
                                               messages=[{"role": "user", "content": prompt}]
                                              )
    
    return response.choices[0].message.content.strip()

'''def generate_article_based_on_topic(topic, corpus, index,lines_number,website):

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    # Find relevant documents based on the topic embedding
    topic_embedding = model.encode([topic])
    D, I = index.search(np.array(topic_embedding), 3)  # Get top 3 related documents

    # Retrieve the content for those documents
    context = "\n".join([corpus[i]["content"] for i in I[0]])

    # Create the prompt for GPT
    prompt = f"""
أنت صحفي عربى محترف في مؤسسة إعلامية بارزة، ومتخصص في كتابة البيانات الإخبارية بلغة عربية فصيحة مع الالتزام بالبيانات والتفاصيل الممنوحة اليك وصياغتها فى صوره بيان مع الالتزام بعدد الاسطر {lines_number} معتمدا فى البيان على البيانات المدخله لك من المستخدم او مواقع رسميه ذات مصادر موثقه مائة باللمائة مع ذكر فى بدايه البيان العنوان الرئيسي و تاريخ اليوم حسب الوطن العربي ثم محتوى البيان ثم كلمة 'للمحررين'  ثم الحول ثم فى نهايه البيان بيانات التواصل من تليفون و ايميل و وسائل التواصل الاجتماعى (ان ذكرت فى التلقين او البيانات المدخله من المستخدم) دون تاليف او تعديل مع ترك مساحتها فارغه اذا لم يتم تحديدها من المستخدم: {topic}.
    استخدم المعلومات التالية كنموذج لكيقية صياغه المبيان :
    {context}
    و الرجوع الى موقعهم الموجود فى  {website} لكتابه حول عنهم من خدمات يقدموها الى من هم 
    """

    # Get response from OpenAI
    response  = openai.chat.completions.create(
      model="gpt-4o-mini",
      store=True,
      messages=[{"role": "user", "content": prompt}]
        )
    
    return response.choices[0].message.content.strip()'''


'''# ✅ 1. إعداد CORS Middleware (اختياري لكن مهم لو عندك Frontend خارجي)
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
    return response'''


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
      

       ''' with open('filtered_corpus.json', 'r', encoding='utf-8') as json_file:
            corpus = json.load(json_file)
        index = faiss.read_index("my_index.index")'''
    
        # Prepare the Arabic prompt
        topic = f"اكتب بيان للشركة {release['organization_name']} حيث محتوى البيان عن {release['about_press']} وبيانات التواصل {release['organization_phone'],release['organization_email'],release['organization_website']} بتاريخ {release['press_date']} واذكر حول الشركه فى النهايه{release['about_organization']} ويكون عدد الاسطر {release['press_lines_number']}"
        context = f"""تكشف أحدث الأبحاث التي أجرتها هاسل ودينسيتي أن أماكن العمل التقنية لا تزال تلحق بأنماط العمل الجديدة
    نموذج بيان صحفي لأي مناسبة
    يمكن تخصيص نموذج بيان صحفي أساسي لأي إعلان تقريبًا. ابدأ باستخدام النموذج أدناه. أثناء الكتابة، املأ النص بين قوسين، ولكن لا تُدرج أسماء الأقسام، فهي مكتوبة بخط مائل.
    [شعار الشركة]
    للنشر الفوري
    العنوان الرئيسي
    [عنوان جذاب وغني بالمعلومات، ويفضل ألا يتجاوز 100 حرف.]
    عنوان فرعي
    [اختياري: عنوان ثانوي موجز يوفر معلومات إضافية.]
    خط التاريخ
    [المدينة، الولاية، التاريخ] – [مقدمة للإعلان.]
    فقرة رئيسية
    [لخّص الأسئلة الخمسة: من، ماذا، متى، أين، ولماذا. قدّم لمحةً موجزةً عن الإعلان.]
    محتوى الجسم
    
    [اشرح الفقرة الافتتاحية بمزيد من التفصيل النقاط الرئيسية وتفاصيل الخلفية وأي سياق ذي صلة. تأكد من أن هذا القسم منظم جيدًا وجذاب.]	
    يقتبس
    [أدرج اقتباسًا من شخصية رئيسية في المنظمة. يُرجى ملاحظة أنه يمكنك أيضًا نشر الاقتباسات في جميع أنحاء صيغة البيان الصحفي.]
    معلومات داعمة
    [قم بتوفير إحصائيات واضحة وموجزة أو اتجاهات أو معلومات أخرى تدعم إعلانك.]
    دعوة إلى العمل
    أخبر القارئ بما تريد منه فعله تاليًا. اجعل دعوته واضحة ومقنعة.
    قالب نموذجي
    قدّم معلومات أساسية عن مؤسستك، بما في ذلك رسالتها وتاريخها وأهم إنجازاتها. اجعلها موجزة ومناسبة لجمهورك. 
    معلومات الاتصال
    [اسم]
    [عنوان]
    [اسم الشركة]
    [رقم التليفون]
    [عنوان البريد الإلكتروني]
    [موقع إلكتروني]
    [مقابض وسائل التواصل الاجتماعي]
    تعزيز الوصول والنتائج مع خبراء البيانات الصحفية
    قبل إتقان تنسيق البيان الصحفي، يمكن للنماذج أن تساعد في ضمان تنظيم إعلاناتك واحترافيتها وتنسيقها بشكل صحيح. لكن الترويج لبيانك الصحفي وضمان وصوله إلى الجمهور المناسب في الوقت المناسب يُعدّ جزءًا مهمًا آخر من عمل متخصصي العلاقات العامة. 
"""
        article = generate_article_based_on_topic(topic,context,release['press_lines_number'],release['organization_website'])
        
       # article = generate_article_based_on_topic(topic, corpus, index,release['press_lines_number'],release['organization_website'])
        
        print(article)
    
        update_data= update_press_release(release['user_id'], release['organization_name'], article)
        print("update_data",update_data)

        
        user = release['user_id'] 
        organization_name = release['organization_name']

        saved_data = update_press_release(user, organization_name, article)
        #print("Processing started...")
        #await asyncio.sleep(5)  # simulate long task
        #print("Processing finished.")
        
        connection.commit()
        connection.close()

    return {"last release": release , "data_condation": saved_data}

if __name__ == "__main__":              
    uvicorn.run(app, host=host, port=port)

