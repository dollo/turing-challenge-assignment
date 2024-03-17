import gradio as gr

from turing_challenge_assignment.modules.chat import dummy_response


gr.ChatInterface(dummy_response).launch()
