#!/usr/bin/env python

import shlex
import subprocess

import gradio as gr

from model import Model
from settings import CACHE_EXAMPLES, MAX_SEED
from utils import randomize_seed_fn


def create_demo(model: Model) -> gr.Blocks:
    subprocess.run(
        shlex.split(
            'wget https://raw.githubusercontent.com/openai/shap-e/d99cedaea18e0989e340163dbaeb4b109fa9e8ec/shap_e/examples/example_data/corgi.png -O corgi.png'
        ))
    examples = ['corgi.png']

    def process_example_fn(image_path: str) -> str:
        return model.run_image(image_path, output_image_size=128)

    with gr.Blocks() as demo:
        with gr.Box():
            image = gr.Image(label='Input image',
                             show_label=False,
                             type='filepath')
            run_button = gr.Button('Run')
            result = gr.Video(label='Result', elem_id='result-2')
            with gr.Accordion('Advanced options', open=False):
                seed = gr.Slider(label='Seed',
                                 minimum=0,
                                 maximum=MAX_SEED,
                                 step=1,
                                 value=0)
                randomize_seed = gr.Checkbox(label='Randomize seed',
                                             value=True)
                guidance_scale = gr.Slider(label='Guidance scale',
                                           minimum=1,
                                           maximum=20,
                                           step=0.1,
                                           value=3.0)
                num_inference_steps = gr.Slider(
                    label='Number of inference steps',
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=64)
                image_size = gr.Slider(label='Image size',
                                       minimum=64,
                                       maximum=256,
                                       step=64,
                                       value=128)
                render_mode = gr.Dropdown(label='Render mode',
                                          choices=['nerf', 'stf'],
                                          value='nerf',
                                          visible=False)

        gr.Examples(examples=examples,
                    inputs=image,
                    outputs=result,
                    fn=process_example_fn,
                    cache_examples=CACHE_EXAMPLES)

        inputs = [
            image,
            seed,
            guidance_scale,
            num_inference_steps,
            image_size,
            render_mode,
        ]

        run_button.click(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
        ).then(
            fn=model.run_image,
            inputs=inputs,
            outputs=result,
        )
    return demo
