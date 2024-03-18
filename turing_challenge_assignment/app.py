from dotenv import load_dotenv
import gradio as gr
import logging

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dotenv
dotenv_loaded = load_dotenv()
logger.info(f"dotenv_loaded: {dotenv_loaded}")


from turing_challenge_assignment.modules.chat import chat_response


# Gradio
gr.ChatInterface(chat_response).launch()
