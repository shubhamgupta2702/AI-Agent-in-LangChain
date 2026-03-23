from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
import requests
import os
from langchain_community.tools import DuckDuckGoSearchRun
from langsmith import Client
from langchain_classic.agents import AgentExecutor, create_react_agent
load_dotenv()
weather_api = os.getenv('WEATHER_STACK_API')
search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city:str) -> str:
  """
  This fetches the weather data from api of a specific city.
  """
  url = f'http://api.weatherstack.com/current?access_key={weather_api}&query={city}'
  response = requests.get(url)
  
  return response.json

# result = search_tool.invoke("todays top news")  -> testing
# print(result)

llm = HuggingFaceEndpoint(
  repo_id="Qwen/Qwen3-Coder-Next",
  task='text-generation'
)

model = ChatHuggingFace(llm=llm)

client = Client()
prompt = client.pull_prompt('hwchase17/react')

agent = create_react_agent(
  llm=model,
  tools=[search_tool, get_weather_data], 
  prompt=prompt
)

agent_executer = AgentExecutor(
  agent=agent,
  tools=[search_tool, get_weather_data],
  verbose=True
)

res = agent_executer.invoke({"input":"plan one day iternary for mathura visit with respect to the weather"})

print(res)