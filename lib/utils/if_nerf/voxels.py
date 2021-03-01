import numpy as np
import trimesh


class VoxelGrid:
    def __init__(self, data, loc=(0., 0., 0.), scale=1):
        assert(data.shape[0] == data.shape[1] == data.shape[2])
        data = np.asarray(data, dtype=np.bool)
        loc = np.asarray(loc)
        self.data = data
        self.loc = loc
        self.scale = scale

    @classmethod
    def from_mesh(cls, mesh, resolution, loc=None, scale=None, method='ray'):
        bounds = mesh.bounds
        # Default location is center
        if loc is None:
            loc = (bounds[0] + bounds[1]) / 2

        # Default scale, scales the mesh to [-0.45, 0.45]^3
        if scale is None:
            scale = (bounds[1] - bounds[0]).max()/0.9

        loc = np.asarray(loc)
        scale = float(scale)

        # Transform mesh
        mesh = mesh.copy()
        mesh.apply_translation(-loc)
        mesh.apply_scale(1/scale)

        # Apply method
        if method == 'ray':
            voxel_data = voxelize_ray(mesh, resolution)
        elif method == 'fill':
            voxel_data = voxelize_fill(mesh, resolution)

        voxels = cls(voxel_data, loc, scale)
        return voxels

    def down_sample(self, factor=2):
        if not (self.resolution % factor) == 0:
            raise ValueError('Resolution must be divisible by factor.')
        new_data = block_reduce(self.data, (factor,) * 3, np.max)
        return VoxelGrid(new_data, self.loc, self.scale)

    def to_mesh(self):
        # Shorthand
        occ = self.data

        # Shape of voxel grid
        nx, ny, nz = occ.shape
        # Shape of corresponding occupancy grid
        grid_shape = (nx + 1, ny + 1, nz + 1)

        # Convert values to occupancies
        occ = np.pad(occ, 1, 'constant')

        # Determine if face present
        f1_r = (occ[:-1, 1:-1, 1:-1] & ~occ[1:, 1:-1, 1:-1])
        f2_r = (occ[1:-1, :-1, 1:-1] & ~occ[1:-1, 1:, 1:-1])
        f3_r = (occ[1:-1, 1:-1, :-1] & ~occ[1:-1, 1:-1, 1:])

        f1_l = (~occ[:-1, 1:-1, 1:-1] & occ[1:, 1:-1, 1:-1])
        f2_l = (~occ[1:-1, :-1, 1:-1] & occ[1:-1, 1:, 1:-1])
        f3_l = (~occ[1:-1, 1:-1, :-1] & occ[1:-1, 1:-1, 1:])

        f1 = f1_r | f1_l
        f2 = f2_r | f2_l
        f3 = f3_r | f3_l

        assert(f1.shape == (nx + 1, ny, nz))
        assert(f2.shape == (nx, ny + 1, nz))
        assert(f3.shape == (nx, ny, nz + 1))

        # Determine if vertex present
        v = np.full(grid_shape, False)

        v[:, :-1, :-1] |= f1
        v[:, :-1, 1:] |= f1
        v[:, 1:, :-1] |= f1
        v[:, 1:, 1:] |= f1

        v[:-1, :, :-1] |= f2
        v[:-1, :, 1:] |= f2
        v[1:, :, :-1] |= f2
        v[1:, :, 1:] |= f2

        v[:-1, :-1, :] |= f3
        v[:-1, 1:, :] |= f3
        v[1:, :-1, :] |= f3
        v[1:, 1:, :] |= f3

        # Calculate indices for vertices
        n_vertices = v.sum()
        v_idx = np.full(grid_shape, -1)
        v_idx[v] = np.arange(n_vertices)

        # Vertices
        v_x, v_y, v_z = np.where(v)
        v_x = v_x / nx - 0.5
        v_y = v_y / ny - 0.5
        v_z = v_z / nz - 0.5
        vertices = np.stack([v_x, v_y, v_z], axis=1)

        # Face indices
        f1_l_x, f1_l_y, f1_l_z = np.where(f1_l)
        f2_l_x, f2_l_y, f2_l_z = np.where(f2_l)
        f3_l_x, f3_l_y, f3_l_z = np.where(f3_l)

        f1_r_x, f1_r_y, f1_r_z = np.where(f1_r)
        f2_r_x, f2_r_y, f2_r_z = np.where(f2_r)
        f3_r_x, f3_r_y, f3_r_z = np.where(f3_r)

        faces_1_l = np.stack([
            v_idx[f1_l_x, f1_l_y, f1_l_z],
            v_idx[f1_l_x, f1_l_y, f1_l_z + 1],
            v_idx[f1_l_x, f1_l_y + 1, f1_l_z + 1],
            v_idx[f1_l_x, f1_l_y + 1, f1_l_z],
        ], axis=1)

        faces_1_r = np.stack([
            v_idx[f1_r_x, f1_r_y, f1_r_z],
            v_idx[f1_r_x, f1_r_y + 1, f1_r_z],
            v_idx[f1_r_x, f1_r_y + 1, f1_r_z + 1],
            v_idx[f1_r_x, f1_r_y, f1_r_z + 1],
        ], axis=1)

        faces_2_l = np.stack([
            v_idx[f2_l_x, f2_l_y, f2_l_z],
            v_idx[f2_l_x + 1, f2_l_y, f2_l_z],
            v_idx[f2_l_x + 1, f2_l_y, f2_l_z + 1],
            v_idx[f2_l_x, f2_l_y, f2_l_z + 1],
        ], axis=1)

        faces_2_r = np.stack([
            v_idx[f2_r_x, f2_r_y, f2_r_z],
            v_idx[f2_r_x, f2_r_y, f2_r_z + 1],
            v_idx[f2_r_x + 1, f2_r_y, f2_r_z + 1],
            v_idx[f2_r_x + 1, f2_r_y, f2_r_z],
        ], axis=1)

        faces_3_l = np.stack([
            v_idx[f3_l_x, f3_l_y, f3_l_z],
            v_idx[f3_l_x, f3_l_y + 1, f3_l_z],
            v_idx[f3_l_x + 1, f3_l_y + 1, f3_l_z],
            v_idx[f3_l_x + 1, f3_l_y, f3_l_z],
        ], axis=1)

        faces_3_r = np.stack([
            v_idx[f3_r_x, f3_r_y, f3_r_z],
            v_idx[f3_r_x + 1, f3_r_y, f3_r_z],
            v_idx[f3_r_x + 1, f3_r_y + 1, f3_r_z],
            v_idx[f3_r_x, f3_r_y + 1, f3_r_z],
        ], axis=1)

        faces = np.concatenate([
            faces_1_l, faces_1_r,
            faces_2_l, faces_2_r,
            faces_3_l, faces_3_r,
        ], axis=0)

        vertices = self.loc + self.scale * vertices
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        return mesh

    @property
    def resolution(self):
        assert(self.data.shape[0] == self.data.shape[1] == self.data.shape[2])
        return self.data.shape[0]

    def contains(self, points):
        nx = self.resolution

        # Rescale bounding box to [-0.5, 0.5]^3
        points = (points - self.loc) / self.scale
        # Discretize points to [0, nx-1]^3
        points_i = ((points + 0.5) * nx).astype(np.int32)
        # i1, i2, i3 have sizes (batch_size, T)
        i1, i2, i3 = points_i[..., 0],  points_i[..., 1],  points_i[..., 2]
        # Only use indices inside bounding box
        mask = (
            (i1 >= 0) & (i2 >= 0) & (i3 >= 0)
            & (nx > i1) & (nx > i2) & (nx > i3)
        )
        # Prevent out of bounds error
        i1 = i1[mask]
        i2 = i2[mask]
        i3 = i3[mask]

        # Compute values, default value outside box is 0
        occ = np.zeros(points.shape[:-1], dtype=np.bool)
        occ[mask] = self.data[i1, i2, i3]

        return occ
