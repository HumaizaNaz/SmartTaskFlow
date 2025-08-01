# SmartTaskFlow

A Python-based multi-agent system using the Gemini API to handle location queries, plant-related questions, and breaking news.

## Features

- Fetches user's location (mocked to Karachi, Pakistan).
- Explains plant concepts like photosynthesis using a dedicated agent.
- Retrieves breaking news headlines (mocked).

## Setup

1. Clone the repository: `git clone https://github.com/HumaizaNaz/SmartTaskFlow.git`
2. Install dependencies: `uv pip install -r requirements.txt`
3. Create a `.env` file with `GEMINI_API_KEY=your-api-key`
4. Run: `uv run main.py`

## Usage

Run the script and ask:

- What is my current location?
- What is photosynthesis?
- Any breaking news?

## Technologies

- Python, Gemini API, agents library, python-dotenv
