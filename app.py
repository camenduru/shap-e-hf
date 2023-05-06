#!/usr/bin/env python

import os

import gradio as gr
import torch

from app_image_to_3d import create_demo as create_demo_image_to_3d
from app_text_to_3d import create_demo as create_demo_text_to_3d
from model import Model

DESCRIPTION = '# [Shap-E](https://github.com/openai/shap-e)'

model = Model()

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        with gr.Tab(label='Text to 3D'):
            create_demo_text_to_3d(model)
        with gr.Tab(label='Image to 3D'):
            create_demo_image_to_3d(model)
demo.queue(api_open=False, max_size=5).launch()
