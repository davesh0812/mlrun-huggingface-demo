import gradio as gr
import requests


def sentiment(text):
    global serving_url
    resp = requests.post(serving_url, json={"text": text})
    return resp.json()


def build_and_launch(url):
    global serving_url
    serving_url = url

    with gr.Blocks() as demo:
        input_box = [
            gr.Textbox(label="Text to analyze", placeholder="Please insert text")
        ]
        output = [
            gr.Textbox(label="Sentiment analysis result"),
            gr.Textbox(label="Sentiment analysis score"),
        ]
        greet_btn = gr.Button("Submit")
        greet_btn.click(
            fn=sentiment,
            inputs=input_box,
            outputs=output,
        )

    demo.launch(share=True)
