from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    function_tool,
)
from dotenv import load_dotenv
import os
from agents.run import RunConfig
from typing import TypedDict

import rich
# ===================================================

load_dotenv()
set_tracing_disabled(disabled=True)
# enable_verbose_stdout_logging()

# ===================================================

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError(
        "GEMINI_API_KEY is not set. Please ensure it is defined in your .env file."
    )
# ===================================================

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
)

# ===================================================
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

config = RunConfig(
    model=model,
    model_provider=external_client,
)
# ===================================================


class Location(TypedDict):
    city: str
    country: str


@function_tool
def get_current_location() -> Location:
    """
    Args: get the current location of the user
    returns: return the city and country of the user"""
    # In a real application, this would fetch the user's actual location.
    return {"city": "Karachi", "country": "Pakistan"}


# ===================================================


@function_tool
def get_breaking_news() -> str:
    """
    Args: get the latest breaking news
    returns: return the latest breaking news headline"""
    # In a real application, this would fetch the latest news from a news API.
    return "Latest Breaking News: Major event happening right now!"


# ===================================================
plant_agent = Agent(
    name="plant_agent",
    instructions="""You are a helpful AI assistant named Plant Agent. Your role is to provide accurate, concise, and clear answers to user queries about plants.""",
    model=model,
)

# ===================================================


main_agent = Agent(
    name="main_agent",
    instructions="""
You are a helpful AI assistant named Main Agent. Your role is to provide accurate, concise, and clear answers to user queries. Follow these guidelines:
1:you can use the plant_agent to answer questions about plants.
2:you can use tool get_current_location to fetch the user's current location.
3:you can use tool get_breaking_news to fetch the latest breaking news.
4:Answer questions directly and factually.
5:If you don't know something, admit it and suggest how the user can find the answer.
6:Use a friendly and professional tone.
7:For complex queries, break down the response into clear sections.
8:If asked for real-time data (e.g., news), fetch the latest information if possible.
""",
    tools=[get_breaking_news, get_current_location],
    handoffs=[plant_agent],
    model=model,
)

# ===================================================

result = Runner.run_sync(
    main_agent,
    """1. What is my current location?
        2. What is photosynthesis?
        3. Any breaking news?""",
    run_config=config,
)

print("=" * 50)
print("Result: ", result.last_agent.name)
rich.print(result.new_items)
print("Result: ", result.final_output)
