import os
import trimesh
import numpy as np
from tqdm import tqdm


def compute_features(mesh):
    features = []

    # Basic geometric features
    features.append(mesh.volume)
    features.append(mesh.area)
    features.append(mesh.volume / mesh.area if mesh.area != 0 else 0)
    features.append(((36 * np.pi) ** (1 / 3) * mesh.volume ** (2 / 3)) / mesh.area if mesh.area != 0 else 0)

    # Convex hull features
    convex_hull = mesh.convex_hull
    features.append(mesh.area / convex_hull.area if convex_hull.area != 0 else 0)
    features.append(mesh.volume / convex_hull.volume if convex_hull.volume != 0 else 0)
    features.append(mesh.area - convex_hull.area)
    features.append(convex_hull.volume - mesh.volume)

    # Distance-based shape features
    center = mesh.centroid
    distances = np.linalg.norm(mesh.vertices - center, axis=1)
    if np.min(distances) > 0:
        features.append(np.max(distances) / np.min(distances))  # aspect ratio
    else:
        features.append(0)
    features.append(np.max(distances) - np.min(distances))       # radial span
    features.append(np.max(distances) ** 3 / mesh.volume if mesh.volume != 0 else 0)
    features.append(np.min(distances) ** 3 / mesh.volume if mesh.volume != 0 else 0)

    # PCA-based eccentricity
    vertices = mesh.vertices
    centered_vertices = vertices - np.mean(vertices, axis=0)
    cov_matrix = np.cov(centered_vertices.T)
    eigenvalues, _ = np.linalg.eigh(cov_matrix)
    long_axis = np.max(eigenvalues)
    short_axis = np.min(eigenvalues)
    eccentricity = (1 - (short_axis / long_axis)) ** 0.5 if long_axis != 0 else 0
    features.append(eccentricity)

    return features


def process_all_meshes(mesh_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    keys, features = [], []

    mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith(".obj") or f.endswith(".ply")]

    for filename in tqdm(mesh_files, desc="Processing Meshes"):
        try:
            mesh_id = int(os.path.splitext(filename)[0])
            mesh = trimesh.load_mesh(os.path.join(mesh_dir, filename))
            fvec = compute_features(mesh)

            keys.append(mesh_id)
            features.append(fvec)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    # Save results
    np.save(os.path.join(output_dir, "keys.npy"), np.array(keys))
    np.save(os.path.join(output_dir, "features.npy"), np.array(features))

    print(f"Finished. Extracted features from {len(keys)} meshes.")


# ===== Run Script =====
if __name__ == "__main__":
    mesh_dir = "/**/**"   # Input folder
    output_dir = "./add_feature"  # Output folder
    process_all_meshes(mesh_dir, output_dir)
