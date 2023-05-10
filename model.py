import tempfile

import imageio
from typing import Union
import numpy as np
import PIL.Image
import torch
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.diffusion.sample import sample_latents
from shap_e.models.download import load_config, load_model
from shap_e.models.nn.camera import (DifferentiableCameraBatch,
                                     DifferentiableProjectiveCamera)
from shap_e.models.transmitter.base import Transmitter, VectorDecoder
from shap_e.util.collections import AttrDict
from shap_e.util.image_util import load_image
from shap_e.rendering.torch_mesh import TorchMesh


# Copied from https://github.com/openai/shap-e/blob/d99cedaea18e0989e340163dbaeb4b109fa9e8ec/shap_e/util/notebooks.py#L15-L42
def create_pan_cameras(size: int,
                       device: torch.device) -> DifferentiableCameraBatch:
    origins = []
    xs = []
    ys = []
    zs = []
    for theta in np.linspace(0, 2 * np.pi, num=20):
        z = np.array([np.sin(theta), np.cos(theta), -0.5])
        z /= np.sqrt(np.sum(z**2))
        origin = -z * 4
        x = np.array([np.cos(theta), -np.sin(theta), 0.0])
        y = np.cross(z, x)
        origins.append(origin)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return DifferentiableCameraBatch(
        shape=(1, len(xs)),
        flat_camera=DifferentiableProjectiveCamera(
            origin=torch.from_numpy(np.stack(origins,
                                             axis=0)).float().to(device),
            x=torch.from_numpy(np.stack(xs, axis=0)).float().to(device),
            y=torch.from_numpy(np.stack(ys, axis=0)).float().to(device),
            z=torch.from_numpy(np.stack(zs, axis=0)).float().to(device),
            width=size,
            height=size,
            x_fov=0.7,
            y_fov=0.7,
        ),
    )


# Copied from https://github.com/openai/shap-e/blob/d99cedaea18e0989e340163dbaeb4b109fa9e8ec/shap_e/util/notebooks.py#L45-L60
@torch.no_grad()
def decode_latent_images(
    xm: Transmitter | VectorDecoder,
    latent: torch.Tensor,
    cameras: DifferentiableCameraBatch,
    rendering_mode: str = 'stf',
):
    decoded = xm.renderer.render_views(
        AttrDict(cameras=cameras),
        params=(xm.encoder if isinstance(xm, Transmitter) else
                xm).bottleneck_to_params(latent[None]),
        options=AttrDict(rendering_mode=rendering_mode,
                         render_with_direction=False),
    )
    arr = decoded.channels.clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    return [PIL.Image.fromarray(x) for x in arr]

@torch.no_grad()
def decode_latent_mesh(
    xm: Union[Transmitter, VectorDecoder],
    latent: torch.Tensor,
) -> TorchMesh:
    decoded = xm.renderer.render_views(
        AttrDict(cameras=create_pan_cameras(2, latent.device)),  # lowest resolution possible
        params=(xm.encoder if isinstance(xm, Transmitter) else xm).bottleneck_to_params(
            latent[None]
        ),
        options=AttrDict(rendering_mode="stf", render_with_direction=False),
    )
    return decoded.raw_meshes[0]

class Model:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.xm = load_model('transmitter', device=self.device)
        self.diffusion = diffusion_from_config(load_config('diffusion'))
        self.model_name = ''
        self.model = None

    def load_model(self, model_name: str) -> None:
        assert model_name in ['text300M', 'image300M']
        if model_name == self.model_name:
            return
        self.model = load_model(model_name, device=self.device)
        self.model_name = model_name

    @staticmethod
    def to_video(frames: list[PIL.Image.Image], fps: int = 5) -> str:
        out_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        writer = imageio.get_writer(out_file.name, format='FFMPEG', fps=fps)
        for frame in frames:
            writer.append_data(np.asarray(frame))
        writer.close()
        return out_file.name

    def run_text(self,
                 prompt: str,
                 seed: int = 0,
                 guidance_scale: float = 15.0,
                 num_steps: int = 64,
                 output_image_size: int = 64,
                 render_mode: str = 'nerf') -> str:
        self.load_model('text300M')

        torch.manual_seed(seed)

        latents = sample_latents(
            batch_size=1,
            model=self.model,
            diffusion=self.diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(texts=[prompt]),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=num_steps,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

        for i, latent in enumerate(latents):
            t = decode_latent_mesh(self.xm, latent).tri_mesh()
            with open(f'../example_txt_mesh_{i}.ply', 'wb') as f:
                t.write_ply(f)
            with open(f'../example_txt_mesh_{i}.obj', 'w') as f:
                t.write_obj(f)

        cameras = create_pan_cameras(output_image_size, self.device)
        frames = decode_latent_images(self.xm,
                                      latents[0],
                                      cameras,
                                      rendering_mode=render_mode)
        return self.to_video(frames)

    def run_image(self,
                  image_path: str,
                  seed: int = 0,
                  guidance_scale: float = 3.0,
                  num_steps: int = 64,
                  output_image_size: int = 64,
                  render_mode: str = 'nerf') -> str:
        self.load_model('image300M')

        torch.manual_seed(seed)

        image = load_image(image_path)

        latents = sample_latents(
            batch_size=1,
            model=self.model,
            diffusion=self.diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(images=[image]),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=num_steps,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

        for i, latent in enumerate(latents):
            t = decode_latent_mesh(self.xm, latent).tri_mesh()
            with open(f'../example_img_mesh_{i}.ply', 'wb') as f:
                t.write_ply(f)
            with open(f'../example_img_mesh_{i}.obj', 'w') as f:
                t.write_obj(f)

        cameras = create_pan_cameras(output_image_size, self.device)
        frames = decode_latent_images(self.xm,
                                      latents[0],
                                      cameras,
                                      rendering_mode=render_mode)
        return self.to_video(frames)
