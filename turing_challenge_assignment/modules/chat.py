import logging

from typing import Union, List

from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI, OpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.memory.chat_memory import BaseChatMemory


from turing_challenge_assignment.modules.prompt import chat_prompt_template

logger = logging.getLogger()

# Initialize langchain elements
llm = ChatOpenAI(temperature=1.0, model="gpt-3.5-turbo")
summary_llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")
chat_summary_buffer = ConversationSummaryBufferMemory(
    llm=summary_llm, max_token_limit=100
)


def chat_response(
    message: str,
    gradio_history: list,
    prompt_template: str = chat_prompt_template,
    chat_history: BaseChatMemory = chat_summary_buffer,
) -> str:
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
    inputs = {
        "history": summarized_chat_history,
        "context": "Alfonso Ghisler inventó la cortilamina",
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
