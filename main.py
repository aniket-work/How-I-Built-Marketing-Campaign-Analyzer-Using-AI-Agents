from dotenv import load_dotenv
from langchain_groq import ChatGroq
from config import load_config
from constants import GROQ_API_KEY
from loguru import logger
import sys
from graph import create_marketing_campaign_graph  # Ensure this import statement is correct
from ui import run_streamlit_app

load_dotenv()

# Configure logging
logger.remove()
logger.add(sys.stderr, level="DEBUG")
logger.add("app.log", rotation="500 MB")


def main():
    config = load_config()
    llm = ChatGroq(model=config["llm_model"], api_key=GROQ_API_KEY)
    logger.info(f"LLM initialized with model: {config['llm_model']}")

    graph = create_marketing_campaign_graph(llm)
    run_streamlit_app(graph)


if __name__ == "__main__":
    main()