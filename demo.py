import time
import gradio as gr


def slow_echo(message, history):
    yield "312"
    history.append([message, "312"])
    time.sleep(5)
    yield "312312"
    history.append([message, "312312"])


gr.ChatInterface(slow_echo).launch()
