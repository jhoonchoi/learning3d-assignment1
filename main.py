import pytorch3d
import torch
import matplotlib.pyplot as plt
import numpy as np
import mcubes

from starter.utils import load_cow_mesh, unproject_depth_image
from starter.dolly_zoom import dolly_zoom
from starter.turntable import render_turntable, save_gif
from starter.camera_transforms import render_textured_cow
from starter.render_generic import load_rgbd_data
"""Entry point for the assignment code submission.

Functions map to each question
"""

"""
1.1 360-degree Renders (5 points)
"""
def spinning_cow(
        obj_path="data/cow.obj", 
        n_frames=72, 
        image_size=256,
        output_file="output/spinning_cow.gif",
):
    # load cow v,f
    cow_v, cow_f = load_cow_mesh(obj_path)

    # reshape
    cow_v = cow_v.unsqueeze(0)
    cow_f = cow_f.unsqueeze(0)

    # generate solid cow texture
    cow_tex = torch.ones_like(cow_v) * torch.tensor([0.7, 0.7, 1])

    # build mesh
    cow_mesh = pytorch3d.structures.Meshes(
        verts=cow_v,
        faces=cow_f,
        textures=pytorch3d.renderer.TexturesVertex(cow_tex),
    )

    # render
    frames = render_turntable(
        cow_mesh, dist=3, elev=0, n_frames=n_frames, image_size=image_size
    )

    # save gif
    save_gif(frames, output_file, fps=30)

"""
1.2 Re-creating the Dolly Zoom (10 points)
"""
def dolly_zoom_render():
    dolly_zoom(
        image_size=256,
        num_frames=60,
        duration=3,
        output_file="output/dolly.gif",
    )

"""
2.1 Constructing a Tetrahedron (5 points)
"""
def construct_tetrahedron(
        n_frames=72, 
        image_size=256,
        output_file="output/tetrahedron.gif",
):
    # define vertices
    v = [
        [1,1,1],
        [-1,-1,1],
        [1,-1,-1],
        [-1,1,-1],
    ]
    vertices = torch.tensor(v, dtype=torch.float32).unsqueeze(0)

    # define faces
    f = [
        [0,1,2],
        [0,3,1],
        [0,2,3],
        [1,3,2],
    ]
    faces = torch.tensor(f).unsqueeze(0)

    # add texture
    textures = torch.ones_like(vertices) * torch.tensor([0.7, 0.7, 1])
    
    # build mesh
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )

    # render
    frames = render_turntable(
        mesh,
        dist=4.5,
        elev=25,
        n_frames=n_frames,
        image_size=image_size,
        flat=True,
    )

    # save gif
    save_gif(frames, output_file, fps=30)

"""
2.2 Constructing a Cube (5 points)
"""
def construct_cube(
        n_frames=72, 
        image_size=256,
        output_file="output/cube.gif",
):
    # define vertices
    v = [
        [1,1,1],
        [1,1,-1],
        [1,-1,1],
        [1,-1,-1],
        [-1,1,1],
        [-1,1,-1],
        [-1,-1,1],
        [-1,-1,-1],
    ]
    vertices = torch.tensor(v, dtype=torch.float32).unsqueeze(0)

    # define faces
    f = [
        [2,6,7],[2,7,3],
        [0,4,6],[0,6,2],
        [1,3,7],[1,7,5],
        [0,2,3],[0,3,1],
        [0,1,5],[0,5,4],
        [4,5,7],[4,7,6],
    ]
    faces = torch.tensor(f).unsqueeze(0)

    # add texture
    textures = torch.ones_like(vertices) * torch.tensor([0.7, 0.7, 1])
    
    # build mesh
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )

    # render
    frames = render_turntable(
        mesh, dist=4,
        elev=25,
        n_frames=n_frames,
        image_size=image_size,
        flat=True
    )

    # save gif
    save_gif(frames, output_file, fps=30)

"""
3. Re-texturing a mesh (10 points)
"""
def color_cow(
        obj_path="data/cow.obj", 
        n_frames=72, 
        image_size=256,
        output_file="output/color_cow.gif",
        color1=torch.tensor([0.5, 0.0, 1.0]),
        color2=torch.tensor([1.0, 0.5, 0.0]),
):
    # load cow v,f
    cow_v, cow_f = load_cow_mesh(obj_path)

    # reshape
    cow_v = cow_v.unsqueeze(0)
    cow_f = cow_f.unsqueeze(0)

    # generate gradient cow texture
    z = cow_v[...,2]
    z_min = z.min()
    z_max = z.max()

    alpha = ((z - z_min) / (z_max - z_min)).unsqueeze(-1)
    color = alpha * color2 + (1 - alpha) * color1

    # build mesh
    cow_mesh = pytorch3d.structures.Meshes(
        verts=cow_v,
        faces=cow_f,
        textures=pytorch3d.renderer.TexturesVertex(color),
    )

    # render
    frames = render_turntable(
        cow_mesh, dist=3, elev=0, n_frames=n_frames, image_size=image_size
    )

    # save gif
    save_gif(frames, output_file, fps=30)

"""
4. Camera Transformations (10 points)
"""
def cow_directions():
    cow_path="data/cow_with_axis.obj"
    cows = []

    # front cow
    cows.append(
        render_textured_cow(
            cow_path=cow_path,
            R_relative=[
                [1,0,0],
                [0,1,0],
                [0,0,1],
            ],
            T_relative=[0,0,0],
        )
    )

    # side cow
    cows.append(
        render_textured_cow(
            cow_path=cow_path,
            R_relative=[
                [0,1,0],
                [-1,0,0],
                [0,0,1],
            ],
            T_relative=[0,0,0],
        )
    )

    # profile cow
    cows.append(
        render_textured_cow(
            cow_path=cow_path,
            R_relative=[
                [0,0,1],
                [0,1,0],
                [-1,0,0],
            ],
            T_relative=[-3,0,3],
        )
    )

    # far cow
    cows.append(
        render_textured_cow(
            cow_path=cow_path,
            R_relative=[
                [1,0,0],
                [0,1,0],
                [0,0,1],
            ],
            T_relative=[0,0,2],
        )
    )

    # slide cow
    cows.append(
        render_textured_cow(
            cow_path=cow_path,
            R_relative=[
                [1,0,0],
                [0,1,0],
                [0,0,1],
            ],
            T_relative=[0.5,-0.5,0],
        )
    )

    for i, cow in enumerate(cows):
        plt.imsave(f"output/transform{i}.jpg", cow)

"""
5.1 Rendering Point Clouds from RGB-D Images (10 points)
"""
def render_plant_points(
        n_frames=36, 
        image_size=256
):
    # keys = [
    #   'rgb1',
    #   'mask1',
    #   'depth1',
    #   'rgb2',
    #   'mask2',
    #   'depth2',
    #   'cameras1',
    #   'cameras2'
    # ]
    data = load_rgbd_data()

    # get points and rgb
    points, rgb = unproject_depth_image(
        torch.tensor(data["rgb1"]),
        torch.tensor(data["mask1"]),
        torch.tensor(data["depth1"]),
        data["cameras1"],
    )

    # build pointcloud
    pointcloud = pytorch3d.structures.Pointclouds(
        points=points.unsqueeze(0),
        features=rgb.unsqueeze(0),
    )

    # render
    frames = render_turntable(
        pointcloud,
        dist=6,
        elev=0,
        n_frames=n_frames,
        image_size=image_size,
        up=((0,-1, 0),),
    )

    # save gif
    save_gif(frames, "output/plant1.gif", fps=15)

    ### image 2 ###
    # get points and rgb
    points2, rgb2 = unproject_depth_image(
        torch.tensor(data["rgb2"]),
        torch.tensor(data["mask2"]),
        torch.tensor(data["depth2"]),
        data["cameras2"],
    )

    # build pointcloud
    pointcloud2 = pytorch3d.structures.Pointclouds(
        points=points2.unsqueeze(0),
        features=rgb2.unsqueeze(0),
    )

    # render
    frames2 = render_turntable(
        pointcloud2, 
        dist=6,
        elev=0,
        n_frames=n_frames,
        image_size=image_size,
        up=((0,-1, 0),),
    )

    # save gif
    save_gif(frames2, "output/plant2.gif", fps=15)

    ### union ###
    union_points = torch.cat([points, points2])
    union_rgb = torch.cat([rgb, rgb2])

    # build pointcloud
    union_pointcloud = pytorch3d.structures.Pointclouds(
        points=union_points.unsqueeze(0),
        features=union_rgb.unsqueeze(0),
    )

    # render
    union_frames = render_turntable(
        union_pointcloud, 
        dist=6,
        elev=0,
        n_frames=n_frames,
        image_size=image_size,
        up=((0,-1, 0),),
    )

    # save gif
    save_gif(union_frames, "output/union_plant.gif", fps=15)

"""
5.2 Parametric Functions (10 + 5 points)
"""
def render_donut(
        num_samples=200,
        n_frames=36, 
        image_size=256,
):
    ### donut ###
    R = 1.0
    r = 0.7
    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, np.pi, num_samples)

    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    # parameterize donut
    x = (R + r * torch.sin(Theta)) * torch.cos(Phi)
    y = (R + r * torch.sin(Theta)) * torch.sin(Phi)
    z = r * torch.cos(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    # build pointcloud
    donut_pointcloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    )

    # render
    donut_frames = render_turntable(
        donut_pointcloud, 
        dist=5,
        elev=30,
        n_frames=n_frames,
        image_size=image_size,
        point_radius=0.015,
    )

    # save gif
    save_gif(donut_frames, "output/donut.gif", fps=15)

def render_cone(
        num_samples=200,
        n_frames=36, 
        image_size=256,
):
    ### cone ###
    H = 2
    R = 1
    phi = torch.linspace(0, 2 * np.pi, num_samples)
    s = torch.linspace(0, R, num_samples)

    # Densely sample phi and theta on a grid
    Phi, S = torch.meshgrid(phi, s)

    # parameterize cone wall
    x = S * torch.cos(Phi)
    y = H * (1 - S / R)
    z = S * torch.sin(Phi)
    wall = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)

    # parameterize cone base
    x = S * torch.cos(Phi)
    y = torch.full_like(S, 0)
    z = S * torch.sin(Phi)
    base = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)

    # combine shapes
    cone = torch.cat([wall, base])

    # center shape
    cone = cone - cone.mean(dim=0)

    # height dependent coloring
    t = cone[:, 1] / H
    color = torch.stack([t, 0.3 * torch.ones_like(t), 1 - t], dim=1)

    # build pointcloud
    cone_pointcloud = pytorch3d.structures.Pointclouds(
        points=[cone], features=[color],
    )

    # render
    cone_frames = render_turntable(
        cone_pointcloud, 
        dist=3.3,
        elev=30,
        n_frames=n_frames,
        image_size=image_size,
        point_radius=0.015,
    )

    # save gif
    save_gif(cone_frames, "output/cone.gif", fps=15)

"""
5.3 Implicit Surfaces (15 + 5 points)
"""
def render_donut_implicit(
        voxel_size=64,
        n_frames=36, 
        image_size=256,
):
    ### donut ###
    min_value = -1.6
    max_value = 1.6

    R = 1.0
    r = 0.4

    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    
    voxels = (torch.sqrt(X**2 + Y**2)-R)**2 + Z**2 - r**2

    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))

    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(textures.unsqueeze(0))

    donut_mesh = pytorch3d.structures.Meshes(
        [vertices], [faces], textures=textures
    )

    # render
    donut_frames = render_turntable(
        donut_mesh, 
        dist=5,
        elev=30,
        n_frames=n_frames,
        image_size=image_size,
        point_radius=0.015,
    )

    # save gif
    save_gif(donut_frames, "output/donut_implicit.gif", fps=15)

def render_cone_implicit(
        voxel_size=64,
        n_frames=36, 
        image_size=256,
):
    ### cone ###
    min_value = -1.2
    max_value = 1.2

    H = 1
    R = 0.6

    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    
    voxels = torch.maximum((torch.sqrt(X**2 + Z**2) - R / H * (H - Y)), -Y)

    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))

    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(textures.unsqueeze(0))

    # shift to center mesh
    vertices = vertices - torch.tensor([0.0, H / 2, 0.0])

    cone_mesh = pytorch3d.structures.Meshes(
        [vertices], [faces], textures=textures
    )

    # render
    cone_frames = render_turntable(
        cone_mesh, 
        dist=2,
        elev=30,
        n_frames=n_frames,
        image_size=image_size,
        point_radius=0.015,
    )

    # save gif
    save_gif(cone_frames, "output/cone_implicit.gif", fps=15)

    return None

"""
(Extra Credit) 6. Do Something Fun (+10 points)
"""
def render_cake(
        num_samples=200,
        n_frames=36, 
        image_size=256,
        slice=0.16,
        height=0.5,
        radius=1.0,
        cake_type="red_velvet",
):
    cake_colors_rgb = {
        "red_velvet": [
            [0.55, 0.10, 0.15],  # crust
            [0.97, 0.96, 0.93],  # top (cream cheese)
            [0.65, 0.12, 0.18],  # sponge
            [0.97, 0.96, 0.93],  # filling (cream cheese)
        ],
        "strawberry_shortcake": [
            [0.93, 0.85, 0.75],
            [1.00, 0.72, 0.78],
            [0.98, 0.92, 0.72],
            [0.85, 0.15, 0.25],
        ],
        "chocolate_fudge": [
            [0.28, 0.16, 0.10],
            [0.36, 0.20, 0.12],
            [0.45, 0.28, 0.16],
            [0.60, 0.42, 0.28],
        ],
        "matcha": [
            [0.45, 0.58, 0.32],
            [0.62, 0.75, 0.45],
            [0.72, 0.82, 0.55],
            [0.98, 0.96, 0.90],
        ],
        "lemon": [
            [0.95, 0.80, 0.40],
            [1.00, 0.95, 0.60],
            [0.98, 0.90, 0.55],
            [0.98, 0.98, 0.85],
        ],
        "black_forest": [
            [0.22, 0.12, 0.10],
            [0.98, 0.97, 0.95],
            [0.30, 0.16, 0.12],
            [0.55, 0.08, 0.12],
        ],
    }
    ### cake ###
    R = float(radius)
    h = float(height)
    alpha = 2 * np.pi * slice

    phi = torch.linspace(0, alpha, num_samples)
    y = torch.linspace(0, h, num_samples)
    s = torch.linspace(0, R, num_samples)

    # crust 
    Phi, Y = torch.meshgrid(phi, y)
    X = R * torch.cos(Phi)
    Z = R * torch.sin(Phi)
    crust = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), dim=1)

    # top
    S, Phi = torch.meshgrid(s, phi)
    X = S * torch.cos(Phi)
    Y = torch.full_like(S, h)
    Z = S * torch.sin(Phi)
    top = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), dim=1)

    # bottom
    S, Phi = torch.meshgrid(s, phi)
    X = S * torch.cos(Phi)
    Y = torch.full_like(S, 0)
    Z = S * torch.sin(Phi)
    bottom = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), dim=1)

    # face1
    S, Y = torch.meshgrid(s, y)
    X = S
    Z = torch.zeros_like(S)
    face1 = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), dim=1)
    
    # face2
    S, Y = torch.meshgrid(s, y)
    X = S * np.cos(alpha)
    Z = S * np.sin(alpha)
    face2 = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), dim=1)

    # full shape
    points = torch.cat([crust, top, bottom, face1, face2])

    # center on the origin so the turntable orbits the slice, not the cake's tip
    points = points - points.mean(dim=0)

    # cake colors: [crust, top, sponge, filling]
    crust_c, top_c, sponge, filling = (torch.tensor(c) for c in cake_colors_rgb[cake_type])
    crust_color = crust_c.expand(len(crust), 3)
    top_color = top_c.expand(len(top), 3)
    bottom_color = sponge.expand(len(bottom), 3)

    y = face1[:, 1] / h
    frac = (y * 3) % 1.0
    is_fill = (frac > 1 - 0.3)[:, None]
    face1_color = torch.where(is_fill, filling, sponge)

    y = face2[:, 1] / h
    frac = (y * 3) % 1.0
    is_fill = (frac > 1 - 0.3)[:, None]
    face2_color = torch.where(is_fill, filling, sponge)

    # full color
    color = torch.cat([crust_color, top_color, bottom_color, face1_color, face2_color])

    # build pointcloud
    donut_pointcloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    )

    # render
    donut_frames = render_turntable(
        donut_pointcloud, 
        dist=3,
        elev=20,
        n_frames=n_frames,
        image_size=image_size,
        point_radius=0.03,
    )

    # save gif
    save_gif(donut_frames, "output/"+cake_type+".gif", fps=15)

    return

"""
(Extra Credit) 7. Sampling Points on Meshes (10 points)
"""
def sample_points_from_mesh(
        mesh,
        n_samples=100,
):
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    areas = mesh.faces_areas_packed()
    face_idx_by_area = torch.multinomial(areas, n_samples, replacement=True)

    alpha = torch.rand(n_samples)
    alpha2 = torch.rand(n_samples)
    alpha1 = 1 - torch.sqrt(alpha)

    w0 = alpha1
    w1 = (1 - alpha1) * alpha2
    w2 = (1 - alpha1) * (1 - alpha2)

    w = torch.stack((w0, w1, w2), dim=1)

    face_verts = verts[faces[face_idx_by_area]]

    return (w[:, :, None] * face_verts).sum(dim=1)

def pc_cow(
        obj_path="data/cow.obj",
        n_samples=100,
        n_frames=36, 
        image_size=256,
        output_file="output/pc_cow.gif",
):
    # load cow v,f
    cow_v, cow_f = load_cow_mesh(obj_path)

    # reshape
    cow_v = cow_v.unsqueeze(0)
    cow_f = cow_f.unsqueeze(0)

    # generate solid cow texture
    cow_tex = torch.ones_like(cow_v) * torch.tensor([0.7, 0.7, 1])

    # build mesh
    cow_mesh = pytorch3d.structures.Meshes(
        verts=cow_v,
        faces=cow_f,
        textures=pytorch3d.renderer.TexturesVertex(cow_tex),
    )

    cow_points = sample_points_from_mesh(cow_mesh,n_samples)

    # generate solid cow texture
    cow_tex = torch.ones_like(cow_points) * torch.tensor([0.7, 0.7, 1])

    # build pointcloud
    cow_pointcloud = pytorch3d.structures.Pointclouds(
        points=cow_points.unsqueeze(0),
        features=cow_tex.unsqueeze(0),
    )

    # render
    frames = render_turntable(
        cow_pointcloud,
        dist=3,
        elev=0,
        n_frames=n_frames,
        image_size=image_size,
    )

    # save gif
    save_gif(frames, output_file, fps=15)

def main():
    # spinning_cow()
    # dolly_zoom_render()
    # construct_tetrahedron()
    # construct_cube()
    # color_cow()
    # cow_directions()
    # render_plant_points()
    # render_donut(num_samples=120)
    # render_cone(num_samples=120)
    # render_cake(num_samples=50, cake_type="red_velvet")
    # render_cake(num_samples=50, cake_type="strawberry_shortcake")
    # render_cake(num_samples=50, cake_type="chocolate_fudge")
    # render_cake(num_samples=50, cake_type="matcha")
    # render_cake(num_samples=50, cake_type="lemon")
    # render_cake(num_samples=50, cake_type="black_forest")
    # render_cake(num_samples=100, cake_type="red_velvet", slice=0.55)
    # render_cake(num_samples=100, cake_type="strawberry_shortcake", slice=0.65)
    # render_cake(num_samples=100, cake_type="chocolate_fudge", height=1, radius=1.3, slice=0.8)
    # render_cake(num_samples=100, cake_type="matcha", slice=0.28)
    # render_cake(num_samples=100, cake_type="lemon", slice=0.45)
    # render_cake(num_samples=100, cake_type="black_forest", height=1, radius=1.3, slice=0.8)
    # render_donut_implicit()
    # render_cone_implicit()
    # pc_cow(n_samples=10, output_file="output/pc_cow_10.gif")
    # pc_cow(n_samples=100, output_file="output/pc_cow_100.gif")
    # pc_cow(n_samples=1000, output_file="output/pc_cow_1000.gif")
    # pc_cow(n_samples=10000, output_file="output/pc_cow_10000.gif")
    return
    

if __name__ == "__main__":
    main()
