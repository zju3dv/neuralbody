import os

# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import numpy as np

import pyrender
import trimesh

colors = [
    (0.5, 0.2, 0.2, 1.0),  # Defalut
    (.7, .5, .5, 1.),  # Pink
    (.7, .7, .6, 1.),  # Neutral
    (.5, .5, .7, 1.),  # Blue
    (.5, .55, .3, 1.),  # capsule
    (.3, .5, .55, 1.),  # Yellow
]


class Renderer(object):

    def __init__(self, focal_length=1000, height=512, width=512, faces=None):
        self.renderer = pyrender.OffscreenRenderer(height, width)
        self.faces = faces
        self.focal_length = focal_length

    def render_multiview(self, vertices, K, R, T, imglist, return_depth=False):
        # List to store rendered scenes
        output_images, output_depths = [], []
        # Need to flip x-axis
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        nViews = len(imglist)
        for nv in range(nViews):
            img = imglist[nv]
            self.renderer.viewport_height = img.shape[0]
            self.renderer.viewport_width = img.shape[1]
            # Create a scene for each image and render all meshes
            scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                   ambient_light=(0.5, 0.5, 0.5))
            camera_pose = np.eye(4)

            if K is None:
                camera_center = np.array([img.shape[1] / 2., img.shape[0] / 2.])
                camera = pyrender.camera.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length, cx=camera_center[0], cy=camera_center[1])
            else:
                camera = pyrender.camera.IntrinsicsCamera(fx=K[nv][0, 0], fy=K[nv][1, 1], cx=K[nv][0, 2], cy=K[nv][1, 2])
            scene.add(camera, pose=camera_pose)
            # Create light source
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
            # for every person in the scene
            if isinstance(vertices, dict):
                for trackId, vert in vertices.items():
                    vert = vert @ R[nv].T + T[nv]
                    mesh = trimesh.Trimesh(vert, self.faces)
                    mesh.apply_transform(rot)
                    trans = [0, 0, 0]

                    material = pyrender.MetallicRoughnessMaterial(
                        metallicFactor=0.2,
                        alphaMode='OPAQUE',
                        baseColorFactor=colors[trackId % len(colors)])
                    mesh = pyrender.Mesh.from_trimesh(
                        mesh,
                        material=material)
                    scene.add(mesh, 'mesh')

                # Use 3 directional lights
                light_pose = np.eye(4)
                light_pose[:3, 3] = np.array([0, -1, 1]) + trans
                scene.add(light, pose=light_pose)
                light_pose[:3, 3] = np.array([0, 1, 1]) + trans
                scene.add(light, pose=light_pose)
                light_pose[:3, 3] = np.array([1, 1, 2]) + trans
                scene.add(light, pose=light_pose)
            else:
                n = 0
                verts = vertices @ R[nv].T + T[nv]
                mesh = trimesh.Trimesh(verts, self.faces)
                mesh.apply_transform(rot)
                trans = [0, 0, 0]

                material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.2,
                    alphaMode='OPAQUE',
                    baseColorFactor=colors[n % len(colors)])
                mesh = pyrender.Mesh.from_trimesh(
                    mesh,
                    material=material)
                scene.add(mesh, 'mesh')

                # Use 3 directional lights
                light_pose = np.eye(4)
                light_pose[:3, 3] = np.array([0, -1, 1]) + trans
                scene.add(light, pose=light_pose)
                light_pose[:3, 3] = np.array([0, 1, 1]) + trans
                scene.add(light, pose=light_pose)
                light_pose[:3, 3] = np.array([1, 1, 2]) + trans
                scene.add(light, pose=light_pose)
            # Alpha channel was not working previously need to check again
            # Until this is fixed use hack with depth image to get the opacity
            color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            # color = color[::-1,::-1]
            # rend_depth = rend_depth[::-1,::-1]
            output_depths.append(rend_depth)
            color = color.astype(np.uint8)
            valid_mask = (rend_depth > 0)[:, :, None]
            output_img = (color[:, :, :3] * valid_mask +
                          (1 - valid_mask) * img)

            output_img = output_img.astype(np.uint8)
            output_images.append(output_img)
        if return_depth:
            return output_images, output_depths
        else:
            return output_images

    def __call__(self, images, vertices, translation, K=None):
        # List to store rendered scenes
        output_images = []
        # Need to flip x-axis
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        # For all iamges
        for i in range(len(images)):
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            self.renderer.viewport_height = img.shape[0]
            self.renderer.viewport_width = img.shape[1]
            verts = vertices[i].detach().cpu().numpy()
            mesh_trans = translation[i].cpu().numpy()
            verts = verts + mesh_trans[:, None, ]
            num_people = verts.shape[0]

            # Create a scene for each image and render all meshes
            scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                   ambient_light=(0.5, 0.5, 0.5))

            # Create camera. Camera will always be at [0,0,0]
            # CHECK If I need to swap x and y
            camera_pose = np.eye(4)

            if K is None:
                camera_center = np.array([img.shape[1] / 2., img.shape[0] / 2.])
                camera = pyrender.camera.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length, cx=camera_center[0], cy=camera_center[1])
            else:
                camera = pyrender.camera.IntrinsicsCamera(fx=K[i][0, 0], fy=K[i][1, 1], cx=K[i][0, 2], cy=K[i][1, 2])
            scene.add(camera, pose=camera_pose)
            # Create light source
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
            # for every person in the scene
            for n in range(num_people):
                mesh = trimesh.Trimesh(verts[n], self.faces)
                mesh.apply_transform(rot)
                trans = 0 * mesh_trans[n]
                trans[0] *= -1
                trans[2] *= -1
                material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.2,
                    alphaMode='OPAQUE',
                    baseColorFactor=colors[n % len(colors)])
                mesh = pyrender.Mesh.from_trimesh(
                    mesh,
                    material=material)
                scene.add(mesh, 'mesh')

                # Use 3 directional lights
                light_pose = np.eye(4)
                light_pose[:3, 3] = np.array([0, -1, 1]) + trans
                scene.add(light, pose=light_pose)
                light_pose[:3, 3] = np.array([0, 1, 1]) + trans
                scene.add(light, pose=light_pose)
                light_pose[:3, 3] = np.array([1, 1, 2]) + trans
                scene.add(light, pose=light_pose)
            # Alpha channel was not working previously need to check again
            # Until this is fixed use hack with depth image to get the opacity
            color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            # color = color[::-1,::-1]
            # rend_depth = rend_depth[::-1,::-1]
            color = color.astype(np.float32) / 255.0
            valid_mask = (rend_depth > 0)[:, :, None]
            output_img = (color[:, :, :] * valid_mask +
                          (1 - valid_mask) * img)
            output_img = np.transpose(output_img, (2, 0, 1))
            output_images.append(output_img)

        return output_images
