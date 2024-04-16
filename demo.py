import gradio as gr

def markdown_response(message, history):
    # Generates a Markdown formatted response
    # For this demo, history is not used, but you must accept it as a parameter.
    text = """
    ```
    copy me !!!
    ```"""
    return text

# Create a ChatInterface, the fn function returns text formatted with Markdown
chat = gr.ChatInterface(
    fn=markdown_response,
    title="Markdown Chat Interface",
    description="This chat interface supports Markdown formatted output."
)

# Launch the interface with share=True to create a shareable public link
chat.launch(share=True)
