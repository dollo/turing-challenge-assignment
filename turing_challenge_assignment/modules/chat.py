import logging

from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI, OpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.memory.chat_memory import BaseChatMemory

from turing_challenge_assignment.modules.prompt import chat_prompt_template
from turing_challenge_assignment.modules.vectorstore import get_db

logger = logging.getLogger()

# Langchain
llm = ChatOpenAI(temperature=1.0, model="gpt-3.5-turbo")
summary_llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")
chat_summary_buffer = ConversationSummaryBufferMemory(
    llm=summary_llm, max_token_limit=100
)

# Chroma
folder_path = "/home/dollo/Workspace/turing-challenge-assignment/documents"
chromadb = get_db(folder_path)


def chat_response(
    message: str,
    gradio_history: list[str],
    prompt_template: str = chat_prompt_template,
    chat_history: BaseChatMemory = chat_summary_buffer,
    db=chromadb,
) -> str:
    """
    Chat response. This function is called to return a response for
    each messaged introduced in the Gradio app.
    """
    # Build chain
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"], template=prompt_template
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Prepare inputs
    logger.info(
        f"chat_history.moving_summary_buffer: {chat_history.moving_summary_buffer}"
    )
    logger.info(
        f"chat_history.chat_memory.messages: {chat_history.chat_memory.messages}"
    )
    summarized_chat_history = [
        chat_history.moving_summary_buffer
    ] + chat_history.chat_memory.messages

    context = db.similarity_search(message)
    logger.info(f"context: {context}")

    inputs = {
        "history": summarized_chat_history,
        "context": context,
        "question": message,
    }

    # Get response
    response = llm_chain.invoke(inputs)
    response = response["text"]

    # Save to history
    chat_history.chat_memory.add_user_message(message)
    chat_history.chat_memory.add_ai_message(response)
    chat_history.prune()
    return response
