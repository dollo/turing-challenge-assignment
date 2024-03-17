from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import logging
from typing import Union

logger = logging.getLogger()

llm = ChatOpenAI(temperature=1.0, model="gpt-3.5-turbo")


def dummy_response(message: str, history: list(Union(HumanMessage, SystemMessage))):
    chat_history = []
    for human, system in history:
        chat_history.append(HumanMessage(content=human))
        chat_history.append(SystemMessage(content=system))
    chat_history.append(HumanMessage(content=message))
    response = llm.invoke(chat_history)
    return response.content
