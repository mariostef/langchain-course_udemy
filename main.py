from dotenv import load_dotenv 
from langchain.agents import create_agent
from langchain.tools import tool
from  langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch



load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-4-hydrogen",
    temperature=0,
)
tools = [TavilySearch(max_results=3, search_depth="basic", include_raw_content=False)]
agent=create_agent(model=llm,tools=tools)

def main() -> None:
    print("Hello")
    result=agent.invoke({"messages":HumanMessage(content="Search for 3 jobs for an ai engineer using langchain in the bay area on linkedin and list their details")})
    print(result)
if __name__ == "__main__":
    main()
