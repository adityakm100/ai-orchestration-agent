from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool

# 1. LLM
llm = ChatGoogleGenerativeAI(        
    model="gemini-2.5-flash",
    temperature=0,
)

# 2. Tool
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# 3. Bind tools
llm_with_tools = llm.bind_tools([get_weather])

# 4. Invoke
resp = llm_with_tools.invoke(
    "What is the weather in San Francisco?"
)

print(resp)
print(resp.content)
