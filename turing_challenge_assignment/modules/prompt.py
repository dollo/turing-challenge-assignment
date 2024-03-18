chat_prompt_template = """
You are a helpful chat assitant. You must answer the questions received paying attention
to the history of the conversation and the context given by external ground-truth sources.

This is the history of the conversation:

'''
{history}
'''

This is the ground-truth context:

'''
{context}
'''

Question: {question}
"""
