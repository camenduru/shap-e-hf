#!/usr/bin/env python

import os

import gradio as gr
import torch

from app_image_to_3d import create_demo as create_demo_image_to_3d
from app_text_to_3d import create_demo as create_demo_text_to_3d
from model import Model

DESCRIPTION = '# [Shap-E](https://github.com/openai/shap-e)'

if (SPACE_ID := os.getenv('SPACE_ID')) is not None:
    DESCRIPTION += f'\n<p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings. <a href="https://huggingface.co/spaces/{SPACE_ID}?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a></p>'
if not torch.cuda.is_available():
    DESCRIPTION += '\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>'

model = Model()

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        with gr.Tab(label='Text to 3D'):
            create_demo_text_to_3d(model)
        with gr.Tab(label='Image to 3D'):
            create_demo_image_to_3d(model)
demo.queue(api_open=False, max_size=5).launch()
