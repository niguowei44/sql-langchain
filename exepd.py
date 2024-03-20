import os

#  csv agent 只能做统计、查询
"""

from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI

google_api_key = os.getenv("GOOGLE_API_KEY")  # 读取系统的api_key
# 设置代理服务器
os.environ["HTTP_PROXY"] = "http://127.0.0.1:10808"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10809"

llm = ChatGoogleGenerativeAI(model="gemini-pro")
agent = create_csv_agent(llm, 'C:\\Users\\administered\\Desktop\\deeplearning\\langc\\Titanic\\Titanic数据集\\train.csv', verbose=True)

print(agent.invoke("whats the square root of the average age?")) # 30
"""

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import pandas as pd

df = pd.read_csv('C:\\Users\\administered\\Desktop\\deeplearning\\langc\\Titanic\\Titanic数据集\\train.csv')
agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
agent.run("how many rows are there?")
