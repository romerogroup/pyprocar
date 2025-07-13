import numpy as np


def get_orthonormal_basis(normal):
    if np.abs(np.dot(normal, [0, 0, 1])) < 0.99:
        v_temp = np.array([0, 0, 1])  # Not parallel to normal
    else:
        v_temp = np.array([0, 1, 0])  # Not parallel to normal
        
    u = np.cross(v_temp,normal).astype(np.float32)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u).astype(np.float32)
    v /= np.linalg.norm(v)  # Ensure normalization
    
    return u, v

def transform_points_to_uv(
                           points:np.ndarray, 
                           u: np.ndarray, 
                           v: np.ndarray, 
                           origin:np.ndarray = np.array([0, 0, 0]),
                           ):
    points_shifted = points - origin
    return np.column_stack(
        [np.dot(points_shifted, u), np.dot(points_shifted, v)]
    )

def find_plane_limits(plane_points:np.ndarray):
    u_limits = plane_points[:, 0].min(), plane_points[:, 0].max()
    v_limits = plane_points[:, 1].min(), plane_points[:, 1].max()
    return u_limits, v_limits

def get_uv_grid(
    grid_interpolation:tuple[int, int],
    u_limits:tuple[float, float],
    v_limits:tuple[float, float],
    ):
    grid_u, grid_v = np.mgrid[
        u_limits[0] : u_limits[1] : complex(0, grid_interpolation[0]),
        v_limits[0] : v_limits[1] : complex(0, grid_interpolation[1]),
    ]
    return grid_u, grid_v

def get_uv_grid_points(grid_u:np.ndarray, grid_v:np.ndarray):
    return np.vstack([grid_u.ravel(), grid_v.ravel()]).T

def get_uv_grid_kpoints(origin:np.ndarray, uv_points:np.ndarray, uv_transformation_matrix:np.ndarray = None, u:np.ndarray = None, v:np.ndarray = None, normal:np.ndarray = None):
    if uv_transformation_matrix is None:
        uv_transformation_matrix = get_uv_transformation_matrix(u, v)
    return origin + uv_points @ uv_transformation_matrix

def get_uv_transformation_matrix(u:np.ndarray, v:np.ndarray):
    return np.vstack([u, v])

def get_transformation_matrix(u:np.ndarray, v:np.ndarray, normal:np.ndarray):
    uv_transformation_matrix = get_uv_transformation_matrix(u, v)
    transformation_matrix = np.vstack([uv_transformation_matrix, normal])
    return transformation_matrix