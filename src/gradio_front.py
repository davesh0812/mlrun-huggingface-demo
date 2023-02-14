import gradio as gr
import requests


def sentiment(text):
    global serving_url, serving_function
    if serving_url is not None:
        resp = requests.post(serving_url, json={"text": text})
        return resp.json()
    else:
        resp = serving_function.invoke(path="/predict", body={"text": text})
        return resp


def build_and_launch(url=None, serving_func=None):
    global serving_url, serving_function
    serving_url = url
    serving_function = serving_func

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
