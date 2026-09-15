import torch
import numpy as np
import pytorch3d.renderer
import pytorch3d.structures
import imageio
from tqdm.auto import tqdm

from starter.utils import get_device, get_mesh_renderer, get_points_renderer

"""
Helper function to render reused turntable view & save gif
"""
def render_turntable(
    obj,
    dist=3.0,
    elev=0.0,
    n_frames=36,
    image_size=256,
    azim_start=0.0,
    up=((0.0, 1.0, 0.0),),
    at=((0.0, 0.0, 0.0),),
    fov=60.0,
    lights=None,
    light_follows_camera=True,
    point_radius=0.01,
    background_color=(1.0, 1.0, 1.0),
    device=None,
    progress=True,
):
    if device is None:
        device = get_device()
    obj = obj.to(device)

    is_pc = isinstance(obj, pytorch3d.structures.Pointclouds)
    if is_pc:
        renderer = get_points_renderer(
            image_size=image_size,
            radius=point_radius,
            background_color=background_color,
            device=device,
        )
    else:
        renderer = get_mesh_renderer(image_size=image_size, device=device)

    azims = torch.linspace(azim_start, azim_start + 360.0, n_frames + 1)[:-1]
    frames = []
    for azim in tqdm(azims, desc="turntable", disable=not progress):
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=dist, elev=elev, azim=float(azim), up=up, at=at
        )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=fov, device=device
        )
        if is_pc:
            rend = renderer(obj, cameras=cameras)
        else:
            if lights is not None:
                frame_lights = lights
            elif light_follows_camera:
                frame_lights = pytorch3d.renderer.PointLights(
                    location=cameras.get_camera_center(), device=device
                )
            else:
                frame_lights = pytorch3d.renderer.PointLights(
                    location=[[0.0, 0.0, -3.0]], device=device
                )
            rend = renderer(obj, cameras=cameras, lights=frame_lights)
        frames.append(to_uint8(rend[0, ..., :3]))
    return frames


def to_uint8(image):
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)


def save_gif(frames, path, fps=15, loop=0):
    duration = 1000 // fps  # ms per frame
    imageio.mimsave(path, frames, duration=duration, loop=loop)
    return path


def hstack_frames(*frame_lists, pad=8, pad_value=255):
    n = len(frame_lists[0])
    assert all(len(f) == n for f in frame_lists), "frame lists must match"
    out = []
    for i in range(n):
        parts = []
        for j, frames in enumerate(frame_lists):
            if j > 0:
                h = frames[i].shape[0]
                parts.append(np.full((h, pad, 3), pad_value, dtype=np.uint8))
            parts.append(frames[i])
        out.append(np.concatenate(parts, axis=1))
    return out
