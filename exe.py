import os

openai_api_key = os.getenv("OPENAI_API_KEY")  # 读取系统的api_key
# print(openai_api_key)
google_api_key = os.getenv("GOOGLE_API_KEY")  # 读取系统的api_key

"""
import openai
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (ChatPromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate)
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)
chat = ChatOpenAI(temperature=0)
template = "You are a helpful assistant that translates english to chinese."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
example_human = HumanMessagePromptTemplate.from_template("Hi")
example_ai = AIMessagePromptTemplate.from_template("你好")
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, example_human, example_ai, human_message_prompt])
chain = LLMChain(llm=chat, prompt=chat_prompt)
message = chain.run(" Please provide a list of domains to limit access using `limit_to_domains`.")
print(message)
"""
"""
import pyttsx3  # 将生成的文本回复转换为语音，并通过扬声器播放

engine = pyttsx3.init()

from langchain_community.chat_models import ChatOpenAI
from langchain.chains import APIChain
from langchain.chains.api import open_meteo_docs
llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)
chain = APIChain.from_llm_and_api_docs(
    llm,
    open_meteo_docs.OPEN_METEO_DOCS,
    verbose=True,
    limit_to_domains=["https://api.open-meteo.com/"],
)
answer = chain.run(
    "江苏常州的天气怎么样?"
)
print(answer)
engine.say(answer)  # 语音回答
engine.runAndWait()
"""


"""
# 访问arxiv 查看论文
from langchain.document_loaders import ArxivLoader  # 文档加载
# 1505.04597 U-Net: Convolutional Networks for Biomedical Image Segmentation
docs = ArxivLoader(query="1505.04597", load_max_docs=10).load()  # load_max_docs 最大文档页数
print(docs[0].metadata)  # meta-information of the Document
print()
print(len(docs))
print()
content_arxiv = docs[0].page_content[:99999]  # all pages of the Document content
"""

"""
使用llm总结论文
3.5turbo 支持 4096 token
"""

"""
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
summary_message = client.chat.completions.create(
    messages=[
        {
            "role": "system",  # 可以加system user assistant
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "帮我总结一下这篇论文内容，请使用500字中文：{}".format(content_arxiv),
        }

    ],
    model="gpt-3.5-turbo-16k",
)
print(summary_message.choices[0].message.content)
# prompt_tokens(输入token)  total_tokens(总token)
print(summary_message.usage.completion_tokens)  # 输出toke
"""

"""
# 保存回答
f = open("message.txt", "w+", encoding="utf-8") :
f.write(summary_message.choices[0].message.content)
f.close()
"""

"""
使用Chain操作sql
"""

"""
from langchain_experimental.sql import SQLDatabaseChain
from langchain import OpenAI, SQLDatabase  # SQLDatabaseChain 移除了

db = SQLDatabase.from_uri("sqlite:///notebooks/Chinook.db")
llm = OpenAI(temperature=0, verbose=True)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
print(db_chain.run("How many employees are there?"))
"""


"""
使用sql agent操作sql
"""




from langchain.chat_models import ChatOpenAI
# langchain0.2之后 使用from langchain_openai import ChatOpenAI

from langchain.agents import *

from langchain.llms import OpenAI
# langchain0.2之后 使用 from langchain_community.llms import OpenAI

from langchain.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
# 数据库设置
db_user = "root"
db_password = os.getenv("db_password")
db_host = "localhost:3306"
db_name = "classicmodels"  # 使用的数据库
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")  # 连接
# 模型设置
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # agent_type=AgentType.OPENAI_FUNCTIONS
    verbose=True
)
# print(agent_executor.run("表的名字是什么"))

print(agent_executor.invoke("根据产品利润的高低，你觉得公司应该多卖哪种产品"))



"""
测试gemini
"""

"""
import google.generativeai as genai
genai.configure(api_key=os.environ["GOOGLE_API_KEY"],transport="rest")
prompt = '帮我写出修改mysql表格内容的sql语句代码'

llm = genai.GenerativeModel('gemini-pro')

response = llm.generate_content(prompt)

print(response.text)
"""

"""
# 模型设置
import getpass
from langchain_google_genai import ChatGoogleGenerativeAI
# 设置代理服务器
os.environ["HTTP_PROXY"] = "http://127.0.0.1:10808"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10809"


if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key")

# ChatGoogleGenerativeAI.(transport="rest")
llm = ChatGoogleGenerativeAI(model="gemini-pro")
# result = llm.invoke("mysql中如何增加一行数据")
# print(result.content)

# 数据库设置
from langchain.sql_database import SQLDatabase
db_user = "root"
db_password = os.getenv("db_password")
db_host = "localhost:3306"
db_name = "classicmodels"
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")  # 连接


from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain.agents import AgentType
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # agent_type=AgentType.OPENAI_FUNCTIONS
    verbose=True
)

print(agent_executor.run("帮我计算每个产品的利润"))
"""





"""
from langchain_community.document_loaders import UnstructuredExcelLoader
loader = UnstructuredExcelLoader("C:\\Users\\administered\\Desktop\\统计排序.xlsx", mode="elements")
docs = loader.load()
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
summary_message = client.chat.completions.create(
    messages=[
        {
            "role": "system",  # 可以加system user assistant
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "帮我找出值最大的人的名字：{}".format(docs),
        }

    ],
    model="gpt-3.5-turbo-16k",
)
print(summary_message.choices[0].message.content)
"""
import speech_recognition as sr  # 监听麦克风输入，实现语音转文本的功能
r = sr.Recognizer()
with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source, duration=1)
    audio = r.listen(source, timeout=5, phrase_time_limit=30)
    audio_text = r.recognize_whisper(audio, language="chinese")
print(audio_text)
"""
test
"""