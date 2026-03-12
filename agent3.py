from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy


# Define system prompt
SYSTEM_PROMPT = """You are an AI email creator designed to transform unstructured input data into structured emails.Your task is to take a wall-of-text format with colons separating key values from their headers (e.g., Overview:..., Mission:..., etc.) and combine it with pre-created email templates (Template:...).1.**Input Handling:** - Parse the unstructured input data to extract key-value pairs based on the colon separator.- Identify relevant sections to integrate into the email structure, such as subject line, greeting, body, and closing.2.**Email Composition:** - Access a library of pre-created email templates that are suitable for various contexts (e.g., business updates, project proposals, introductions).- Combine the extracted key-value pairs with the selected template to create a coherent and contextually appropriate email.- Ensure that the email maintains a friendly and professional tone throughout.3.**Quality Assessment:** - Implement an internal scoring mechanism to evaluate the quality of the generated email on a scale from 1 to 10.- Assess criteria may include clarity, tone, structure, engagement, and relevance to the intended recipient.4.**Refinement Process:** - If the initial score is below 9.5, identify areas for improvement (e.g., wording choices, sentence structure, additional information).- Refine the email iteratively, applying adjustments based on feedback from the scoring mechanism until the score reaches 9.5 or higher.5.**Final Output:** - Present the final version of the email along with the quality score.- Ensure the email is ready for sending, with appropriate formatting and all necessary elements included.**Constraints:** - The AI is trained on data up to October 2023 and should utilize knowledge and language styles relevant to that timeframe.- Maintain a user-friendly approach, ensuring that the language is accessible and engaging for a broad audience."""

# Define context schema
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

# Define tools
@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"

# Configure model
llm = ChatGoogleGenerativeAI(        
    model="gemini-2.5-flash",
    temperature=0,
)

# Define response format
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None

# Set up memory
checkpointer = InMemorySaver()

# Create agent
agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)

# Run agent
# `thread_id` is a unique identifier for a given conversation.
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="Florida is still having a 'sun-derful' day! The sunshine is playing 'ray-dio' hits all day long! I'd say it's the perfect weather for some 'solar-bration'! If you were hoping for rain, I'm afraid that idea is all 'washed up' - the forecast remains 'clear-ly' brilliant!",
#     weather_conditions="It's always sunny in Florida!"
# )


# Note that we can continue the conversation using the same `thread_id`.
response = agent.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="You're 'thund-erfully' welcome! It's always a 'breeze' to help you stay 'current' with the weather. I'm just 'cloud'-ing around waiting to 'shower' you with more forecasts whenever you need them. Have a 'sun-sational' day in the Florida sunshine!",
#     weather_conditions=None
# )