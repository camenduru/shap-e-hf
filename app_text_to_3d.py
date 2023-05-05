#!/usr/bin/env python

import gradio as gr

from model import Model
from settings import CACHE_EXAMPLES, MAX_SEED
from utils import randomize_seed_fn


def create_demo(model: Model) -> gr.Blocks:
    examples = [
        'A chair that looks like an avocado',
        'An airplane that looks like a banana',
        'A spaceship',
        'A birthday cupcake',
        'A chair that looks like a tree',
        'A green boot',
        'A penguin',
        'Ube ice cream cone',
        'A bowl of vegetables',
    ]

    def process_example_fn(prompt: str) -> str:
        return model.run_text(prompt, output_image_size=128)

    with gr.Blocks() as demo:
        with gr.Box():
            with gr.Row(elem_id='prompt-container'):
                prompt = gr.Text(
                    label='Prompt',
                    show_label=False,
                    max_lines=1,
                    placeholder='Enter your prompt').style(container=False)
                run_button = gr.Button('Run').style(full_width=False)
            result = gr.Video(label='Result', elem_id='result-1')
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
                                           value=15.0)
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
                    inputs=prompt,
                    outputs=result,
                    fn=process_example_fn,
                    cache_examples=CACHE_EXAMPLES)

        inputs = [
            prompt,
            seed,
            guidance_scale,
            num_inference_steps,
            image_size,
            render_mode,
        ]
        prompt.submit(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
        ).then(
            fn=model.run_text,
            inputs=inputs,
            outputs=result,
        )

        run_button.click(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
        ).then(
            fn=model.run_text,
            inputs=inputs,
            outputs=result,
        )
    return demo
