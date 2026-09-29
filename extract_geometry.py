"""
LAM Geometry Bridge — extract_geometry.py

Executes the LAM-20K model to extract canonical 3D geometry from a single
facial image, WITHOUT rendering the Gaussian avatar.

This script intercepts the LAM pipeline after forward_gs() — the step that
constructs the canonical GaussianModel — and before any rendering.

Extracted data:
    - xyz:              canonical 3D positions of Gaussian points
    - offset:           LAM-predicted offsets from FLAME template
    - flame_shape:      FLAME shape parameters (betas)
    - flame_expression: FLAME expression parameters
    - flame_pose:       FLAME pose parameters (jaw, rotation, neck, eyes)
    - faces:            FLAME mesh face indices (if available)
    - lbs_weights:      Linear Blend Skinning weights (if available)

Usage:
    python extract_geometry.py --image path/to/face.jpg [--debug] [--output output/]

Requirements:
    - LAM repository cloned at ./lam_repo/
    - LAM-20K checkpoint at ./model_zoo/lam_models/ (model.safetensors)
    - Python packages: torch, safetensors, omegaconf, einops, accelerate, diffusers
    - NO CUDA required (CPU inference supported, slow but one-shot)
    - NO pytorch3d required (mocked for geometry-only extraction)
    - NO diff_gaussian_rasterization required (rendering is disabled)
"""

import os
import sys
import time
import argparse
import types
import importlib
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 1. MOCK UNAVAILABLE MODULES
#    pytorch3d and diff_gaussian_rasterization are not needed for geometry
#    extraction, but the LAM source imports them at the top level.
#    We inject lightweight stubs before importing LAM.
# ---------------------------------------------------------------------------

def _install_mock_modules():
    """Create minimal mock modules for dependencies we don't need."""

    # --- diff_gaussian_rasterization (renderer — not needed) ---
    for mod_name in [
        "diff_gaussian_rasterization",
        "diff_gaussian_rasterization_wda",
    ]:
        mock = types.ModuleType(mod_name)
        mock.GaussianRasterizationSettings = type("GaussianRasterizationSettings", (), {})
        mock.GaussianRasterizer = type("GaussianRasterizer", (), {})
        sys.modules[mod_name] = mock

    # --- pytorch3d (only matrix_to_quaternion and a few utilities are used) ---
    def _matrix_to_quaternion(matrix):
        """
        Pure-PyTorch implementation of pytorch3d.transforms.matrix_to_quaternion.
        Converts a batch of rotation matrices [*, 3, 3] to quaternions [*, 4] (w, x, y, z).
        """
        if matrix.shape[-2:] != (3, 3):
            raise ValueError(f"Expected rotation matrices of shape (*, 3, 3), got {matrix.shape}")

        batch_shape = matrix.shape[:-2]
        m = matrix.reshape(-1, 3, 3)
        batch_size = m.shape[0]

        # Shepperd's method
        trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]

        qw = torch.zeros(batch_size, device=matrix.device, dtype=matrix.dtype)
        qx = torch.zeros_like(qw)
        qy = torch.zeros_like(qw)
        qz = torch.zeros_like(qw)

        # Case 1: trace > 0
        s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2  # s = 4*qw
        mask = trace > 0
        qw = torch.where(mask, 0.25 * s, qw)
        qx = torch.where(mask, (m[:, 2, 1] - m[:, 1, 2]) / s, qx)
        qy = torch.where(mask, (m[:, 0, 2] - m[:, 2, 0]) / s, qy)
        qz = torch.where(mask, (m[:, 1, 0] - m[:, 0, 1]) / s, qz)

        # Case 2: m[0,0] is max diagonal
        cond2 = (~mask) & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
        s2 = torch.sqrt(torch.clamp(1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], min=1e-10)) * 2
        qw = torch.where(cond2, (m[:, 2, 1] - m[:, 1, 2]) / s2, qw)
        qx = torch.where(cond2, 0.25 * s2, qx)
        qy = torch.where(cond2, (m[:, 0, 1] + m[:, 1, 0]) / s2, qy)
        qz = torch.where(cond2, (m[:, 0, 2] + m[:, 2, 0]) / s2, qz)

        # Case 3: m[1,1] is max diagonal
        cond3 = (~mask) & (~cond2) & (m[:, 1, 1] > m[:, 2, 2])
        s3 = torch.sqrt(torch.clamp(1.0 + m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2], min=1e-10)) * 2
        qw = torch.where(cond3, (m[:, 0, 2] - m[:, 2, 0]) / s3, qw)
        qx = torch.where(cond3, (m[:, 0, 1] + m[:, 1, 0]) / s3, qx)
        qy = torch.where(cond3, 0.25 * s3, qy)
        qz = torch.where(cond3, (m[:, 1, 2] + m[:, 2, 1]) / s3, qz)

        # Case 4: m[2,2] is max diagonal
        cond4 = (~mask) & (~cond2) & (~cond3)
        s4 = torch.sqrt(torch.clamp(1.0 + m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1], min=1e-10)) * 2
        qw = torch.where(cond4, (m[:, 1, 0] - m[:, 0, 1]) / s4, qw)
        qx = torch.where(cond4, (m[:, 0, 2] + m[:, 2, 0]) / s4, qx)
        qy = torch.where(cond4, (m[:, 1, 2] + m[:, 2, 1]) / s4, qy)
        qz = torch.where(cond4, 0.25 * s4, qz)

        quaternions = torch.stack([qw, qx, qy, qz], dim=-1)
        return quaternions.reshape(*batch_shape, 4)


    def _estimate_pointcloud_normals(pcl, neighborhood_size=50, disambiguate_directions=True):
        """Stub: returns zero normals. Not needed for geometry extraction."""
        pts = pcl if isinstance(pcl, torch.Tensor) else pcl.points_packed()
        return torch.zeros_like(pts)


    # Build module hierarchy
    for mod_path in [
        "pytorch3d",
        "pytorch3d.transforms",
        "pytorch3d.ops",
        "pytorch3d.ops.points_normals",
        "pytorch3d.ops.interp_face_attrs",
        "pytorch3d.structures",
        "pytorch3d.renderer",
        "pytorch3d.renderer.blending",
    ]:
        if mod_path not in sys.modules:
            sys.modules[mod_path] = types.ModuleType(mod_path)

    sys.modules["pytorch3d.transforms"].matrix_to_quaternion = _matrix_to_quaternion
    sys.modules["pytorch3d.ops.points_normals"].estimate_pointcloud_normals = _estimate_pointcloud_normals

    # Mock interpolate_face_attributes
    sys.modules["pytorch3d.ops.interp_face_attrs"].interpolate_face_attributes = lambda *a, **kw: None

    # Mock structures
    sys.modules["pytorch3d.structures"].Meshes = type("Meshes", (), {"__init__": lambda self, *a, **kw: None})
    sys.modules["pytorch3d.structures"].Pointclouds = type("Pointclouds", (), {"__init__": lambda self, *a, **kw: None})

    # Mock renderer classes
    renderer_mod = sys.modules["pytorch3d.renderer"]
    for cls_name in [
        "AmbientLights", "PerspectiveCameras", "SoftSilhouetteShader",
        "SoftPhongShader", "RasterizationSettings", "MeshRenderer",
        "MeshRendererWithFragments", "MeshRasterizer", "TexturesVertex",
    ]:
        setattr(renderer_mod, cls_name, type(cls_name, (), {"__init__": lambda self, *a, **kw: None}))

    blending_mod = sys.modules["pytorch3d.renderer.blending"]
    blending_mod.BlendParams = type("BlendParams", (), {"__init__": lambda self, *a, **kw: None})
    blending_mod.softmax_rgb_blend = lambda *a, **kw: None

    # --- chumpy (sometimes needed by FLAME model loading) ---
    if "chumpy" not in sys.modules:
        mock_chumpy = types.ModuleType("chumpy")
        mock_chumpy.array = np.array
        sys.modules["chumpy"] = mock_chumpy

    print("[mock] Installed mock modules: pytorch3d, diff_gaussian_rasterization, chumpy")


# ---------------------------------------------------------------------------
# 2. MAIN GEOMETRY EXTRACTION
# ---------------------------------------------------------------------------

def extract_lam_geometry(
    image_path: str,
    lam_repo_path: str = "./lam_repo",
    config_path: str = None,
    checkpoint_path: str = None,
    human_model_path: str = None,
    device: str = "cpu",
    debug: bool = False,
    output_dir: str = "./output",
):
    """
    Execute LAM reconstruction up to canonical geometry, WITHOUT rendering.

    Parameters
    ----------
    image_path : str
        Path to a facial photograph.
    lam_repo_path : str
        Path to the cloned LAM repository.
    config_path : str
        Path to the LAM config YAML (defaults to lam-20k-8gpu.yaml).
    checkpoint_path : str
        Path to the model.safetensors checkpoint.
    human_model_path : str
        Path to FLAME model files.
    device : str
        "cpu" or "cuda".
    debug : bool
        If True, generate visualization images.
    output_dir : str
        Where to save extracted geometry.

    Returns
    -------
    dict with keys: xyz, offset, flame_shape, flame_expression, flame_pose, etc.
    """

    # --- Install mocks BEFORE any LAM imports ---
    _install_mock_modules()

    # --- Add LAM repo to path ---
    lam_repo = Path(lam_repo_path).resolve()
    if str(lam_repo) not in sys.path:
        sys.path.insert(0, str(lam_repo))
    print(f"[path] LAM repo: {lam_repo}")

    # --- Resolve default paths ---
    if config_path is None:
        config_path = str(lam_repo / "configs" / "inference" / "lam-20k-8gpu.yaml")
    if human_model_path is None:
        human_model_path = str(lam_repo / "model_zoo" / "human_parametric_models")
    if checkpoint_path is None:
        # Try common locations
        for candidate in [
            lam_repo / "model_zoo" / "lam_models" / "model.safetensors",
            Path("./model_zoo/lam_models/model.safetensors"),
            Path("./model_zoo/model.safetensors"),
        ]:
            if candidate.exists():
                checkpoint_path = str(candidate)
                break

    # --- Validate paths ---
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    print(f"\n{'='*60}")
    print(f"  LAM GEOMETRY BRIDGE — Canonical Geometry Extraction")
    print(f"{'='*60}")
    print(f"  Image:      {image_path}")
    print(f"  Config:     {config_path}")
    print(f"  Checkpoint: {checkpoint_path or 'NOT FOUND'}")
    print(f"  FLAME:      {human_model_path}")
    print(f"  Device:     {device}")
    print(f"  Debug:      {debug}")
    print(f"  Output:     {output_dir}")
    print(f"{'='*60}\n")

    # --- Load config ---
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(config_path)
    model_cfg = dict(cfg.model)
    model_cfg["human_model_path"] = human_model_path

    # --- Check if FLAME model files exist ---
    flame_pkl = Path(human_model_path) / "flame_assets" / "flame" / "flame2023.pkl"
    if not flame_pkl.exists():
        print(f"\n[ERROR] FLAME model file not found: {flame_pkl}")
        print(f"[ERROR] You need to download the FLAME model files.")
        print(f"[ERROR] Expected location: {human_model_path}/flame_assets/flame/")
        print(f"[ERROR]")
        print(f"[ERROR] Required files:")
        print(f"[ERROR]   - flame2023.pkl")
        print(f"[ERROR]   - landmark_embedding_with_eyes.npy")
        print(f"[ERROR]   - head_template_mesh.obj")
        print(f"[ERROR]   - FLAME_masks.pkl")
        print(f"[ERROR]")
        print(f"[ERROR] Download from: https://huggingface.co/3DAIGC/LAM")
        print(f"[ERROR] Or run: huggingface-cli download 3DAIGC/LAM --local-dir {lam_repo}")
        return None

    # --- Check if checkpoint exists ---
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        print(f"\n[ERROR] LAM-20K checkpoint not found.")
        print(f"[ERROR] Download from: https://huggingface.co/3DAIGC/LAM")
        print(f"[ERROR] Place model.safetensors in: ./model_zoo/lam_models/")
        return None

    # --- Disable torch.compile for CPU / compatibility ---
    # The @torch.compile decorator on forward_latent_points causes issues on CPU
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    # --- Import LAM model ---
    print("[1/6] Importing LAM model...")
    t0 = time.time()
    from lam.models.modeling_lam import ModelLAM
    print(f"       Import took {time.time() - t0:.1f}s")

    # --- Create model ---
    print("[2/6] Creating ModelLAM...")
    t0 = time.time()
    model = ModelLAM(**model_cfg)
    print(f"       Model created in {time.time() - t0:.1f}s")

    # --- Load checkpoint ---
    print("[3/6] Loading checkpoint...")
    t0 = time.time()
    from safetensors.torch import load_file
    ckpt = load_file(checkpoint_path, device="cpu")
    state_dict = model.state_dict()
    loaded, skipped = 0, 0
    for k, v in ckpt.items():
        if k in state_dict and state_dict[k].shape == v.shape:
            state_dict[k].copy_(v)
            loaded += 1
        else:
            skipped += 1
    print(f"       Loaded {loaded} params, skipped {skipped} in {time.time() - t0:.1f}s")

    model = model.to(device)
    model.eval()
    print(f"       LAM checkpoint loaded ✓")

    # --- Preprocess image ---
    print("[4/6] Preprocessing image...")
    t0 = time.time()
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    print(f"       Image size: {img_np.shape}")

    # Resize to expected size (512x512 as per LAM config, aspect-preserving with center crop)
    src_size = 512  # default LAM source size
    h, w = img_np.shape[:2]
    scale = src_size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    import cv2
    img_resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center crop to square
    ch, cw = new_h // 2, new_w // 2
    half = src_size // 2
    img_crop = img_resized[ch - half:ch + half, cw - half:cw + half]

    # Normalize to [0, 1] then to tensor [1, 3, H, W]
    img_tensor = torch.from_numpy(img_crop).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)  # [1, 3, 512, 512]
    print(f"       Preprocessed to {img_tensor.shape} in {time.time() - t0:.1f}s")

    # --- Create minimal FLAME params ---
    # For canonical geometry extraction, we need neutral pose FLAME params
    print("[5/6] Extracting canonical geometry...")
    t0 = time.time()

    shape_dim = model_cfg.get("shape_param_dim", 10)
    expr_dim = model_cfg.get("expr_param_dim", 10)

    # Neutral pose: all zeros
    flame_params = {
        "betas": torch.zeros(1, shape_dim, device=device),
        "expr": torch.zeros(1, 1, expr_dim, device=device),        # [B, Nv, expr_dim]
        "rotation": torch.zeros(1, 1, 3, device=device),            # [B, Nv, 3]
        "neck_pose": torch.zeros(1, 1, 3, device=device),           # [B, Nv, 3]
        "jaw_pose": torch.zeros(1, 1, 3, device=device),            # [B, Nv, 3]
        "eyes_pose": torch.zeros(1, 1, 6, device=device),           # [B, Nv, 6]
        "translation": torch.zeros(1, 1, 3, device=device),         # [B, Nv, 3]
    }

    with torch.no_grad():
        # Step 1: Get query points from FLAME (canonical vertices)
        query_points = None
        if model.latent_query_points_type.startswith("e2e_flame"):
            query_points, flame_params = model.renderer.get_query_points(
                flame_params, device=torch.device(device)
            )
            print(f"       FLAME query points: {query_points.shape}")

        # Step 2: Encode image → latent features
        latent_points, image_feats = model.forward_latent_points(
            img_tensor, camera=None, query_points=query_points
        )
        print(f"       Latent features: {latent_points.shape}")

        # Step 3: Decode to Gaussian attributes (INTERCEPT HERE — no rendering)
        from einops import rearrange
        image_feats_bchw = rearrange(
            image_feats, "b (h w) c -> b c h w",
            h=int(image_feats.shape[1] ** 0.5)
        )
        gs_model_list, query_pts_out, flame_data_out, _ = model.renderer.forward_gs(
            gs_hidden_features=latent_points,
            query_points=query_points,
            flame_data=flame_params,
            additional_features={
                "image_feats": image_feats,
                "image": img_tensor,
                "image_feats_bchw": image_feats_bchw,
            },
        )
        print(f"       Canonical Gaussian models: {len(gs_model_list)}")

    inference_time = time.time() - t0

    # --- Extract geometry from the GaussianModel ---
    gs = gs_model_list[0]  # single image → single model

    geometry = {
        "xyz": gs.xyz.detach().cpu().numpy() if gs.xyz is not None else None,
        "offset": gs.offset.detach().cpu().numpy() if gs.offset is not None else None,
        "opacity": gs.opacity.detach().cpu().numpy() if gs.opacity is not None else None,
        "rotation": gs.rotation.detach().cpu().numpy() if gs.rotation is not None else None,
        "scaling": gs.scaling.detach().cpu().numpy() if gs.scaling is not None else None,
        "shs_shape": list(gs.shs.shape) if gs.shs is not None else None,
    }

    # Extract FLAME params
    flame_info = {}
    for k, v in flame_data_out.items():
        if isinstance(v, torch.Tensor):
            flame_info[f"flame_{k}"] = v.detach().cpu().numpy()

    # Extract FLAME model data if available
    flame_model_info = {}
    try:
        flame_model = model.renderer.flame_model
        if hasattr(flame_model, "faces"):
            flame_model_info["faces"] = flame_model.faces.detach().cpu().numpy() if isinstance(flame_model.faces, torch.Tensor) else np.array(flame_model.faces)
        if hasattr(flame_model, "lbs_weights"):
            flame_model_info["lbs_weights"] = flame_model.lbs_weights.detach().cpu().numpy() if isinstance(flame_model.lbs_weights, torch.Tensor) else None
    except Exception as e:
        print(f"       [WARN] Could not extract FLAME model data: {e}")

    if query_pts_out is not None:
        geometry["query_points"] = query_pts_out.detach().cpu().numpy()

    # --- Print results ---
    print(f"\n{'='*60}")
    print(f"  EXTRACTION RESULTS")
    print(f"{'='*60}")
    print(f"  Canonical points:   {geometry['xyz'].shape if geometry['xyz'] is not None else 'N/A'}")
    print(f"  XYZ range:          [{geometry['xyz'].min():.4f}, {geometry['xyz'].max():.4f}]" if geometry['xyz'] is not None else "  XYZ: N/A")
    print(f"  Offsets:            {geometry['offset'].shape if geometry['offset'] is not None else 'N/A'}")
    print(f"  Offset range:       [{geometry['offset'].min():.6f}, {geometry['offset'].max():.6f}]" if geometry['offset'] is not None else "  Offsets: N/A")
    print(f"  Opacity:            {geometry['opacity'].shape if geometry['opacity'] is not None else 'N/A'}")
    print(f"  Rotation (quats):   {geometry['rotation'].shape if geometry['rotation'] is not None else 'N/A'}")
    print(f"  Scaling:            {geometry['scaling'].shape if geometry['scaling'] is not None else 'N/A'}")
    print(f"  SHS shape:          {geometry['shs_shape']}")
    print(f"  Query points:       {geometry.get('query_points', np.array([])).shape if geometry.get('query_points') is not None else 'N/A'}")
    print(f"  FLAME params:       {list(flame_info.keys())}")
    print(f"  FLAME model data:   {list(flame_model_info.keys())}")
    print(f"  Device:             {device}")
    print(f"  Inference time:     {inference_time:.1f}s")
    print(f"  Avatar rendering:   DISABLED ✓")
    print(f"{'='*60}")

    # --- Save to file ---
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(image_path).stem
    save_path = os.path.join(output_dir, f"{stem}_geometry.npz")

    save_data = {}
    for k, v in geometry.items():
        if v is not None and isinstance(v, np.ndarray):
            save_data[k] = v
    for k, v in flame_info.items():
        if v is not None and isinstance(v, np.ndarray):
            save_data[k] = v
    for k, v in flame_model_info.items():
        if v is not None and isinstance(v, np.ndarray):
            save_data[k] = v

    np.savez_compressed(save_path, **save_data)
    print(f"\n  Saved: {save_path}")
    print(f"  Keys:  {list(save_data.keys())}")

    # --- Debug visualization ---
    if debug:
        _generate_debug_visualizations(geometry, flame_model_info, img_crop, output_dir, stem)

    return {**geometry, **flame_info, **flame_model_info}


def _generate_debug_visualizations(geometry, flame_model_info, img_crop, output_dir, stem):
    """Generate debug images: 3D scatter plot and 2D projection overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz = geometry.get("xyz")
    if xyz is None:
        print("  [debug] No XYZ data to visualize")
        return

    # --- 1. 3D Scatter Plot ---
    fig = plt.figure(figsize=(12, 5))

    # Front view
    ax1 = fig.add_subplot(131, projection="3d")
    step = max(1, len(xyz) // 5000)  # subsample for speed
    pts = xyz[::step]
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.3, c=pts[:, 2], cmap="viridis", alpha=0.6)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title(f"Front ({len(xyz)} pts)")
    ax1.view_init(elev=0, azim=0)

    # Top view
    ax2 = fig.add_subplot(132, projection="3d")
    ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.3, c=pts[:, 2], cmap="viridis", alpha=0.6)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title("Top")
    ax2.view_init(elev=90, azim=0)

    # Side view
    ax3 = fig.add_subplot(133, projection="3d")
    ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.3, c=pts[:, 2], cmap="viridis", alpha=0.6)
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.set_title("Side")
    ax3.view_init(elev=0, azim=90)

    plt.suptitle(f"LAM Canonical Geometry — {stem}", fontsize=14)
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, f"{stem}_geometry_3d.png")
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [debug] 3D scatter: {scatter_path}")

    # --- 2. Simple projection on image ---
    # Orthographic projection (XY plane) for quick verification
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Original image
    axes[0].imshow(img_crop)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Projected points on image
    axes[1].imshow(img_crop, alpha=0.5)
    # Simple orthographic: map XYZ to pixel coords
    # FLAME canonical space: X is left-right, Y is up-down, Z is depth
    # Need to scale to image coordinates
    h, w = img_crop.shape[:2]
    x_range = xyz[:, 0].max() - xyz[:, 0].min()
    y_range = xyz[:, 1].max() - xyz[:, 1].min()
    scale = min(w, h) * 0.8 / max(x_range, y_range)

    px = (xyz[::step, 0] - xyz[:, 0].mean()) * scale + w / 2
    py = -(xyz[::step, 1] - xyz[:, 1].mean()) * scale + h / 2  # flip Y

    # Color by depth
    depth = xyz[::step, 2]
    axes[1].scatter(px, py, s=0.2, c=depth, cmap="coolwarm", alpha=0.7)
    axes[1].set_xlim(0, w)
    axes[1].set_ylim(h, 0)
    axes[1].set_title(f"Canonical Geometry Projection ({len(xyz)} pts)")
    axes[1].axis("off")

    plt.suptitle(f"LAM Geometry — {stem}", fontsize=14)
    plt.tight_layout()
    proj_path = os.path.join(output_dir, f"{stem}_geometry_projection.png")
    plt.savefig(proj_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [debug] Projection: {proj_path}")


# ---------------------------------------------------------------------------
# 3. CLI ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LAM Geometry Bridge — extract canonical 3D geometry from a face image"
    )
    parser.add_argument("--image", required=True, help="Path to a facial image")
    parser.add_argument("--lam-repo", default="./lam_repo", help="Path to cloned LAM repository")
    parser.add_argument("--config", default=None, help="Path to LAM config YAML")
    parser.add_argument("--checkpoint", default=None, help="Path to model.safetensors")
    parser.add_argument("--human-model-path", default=None, help="Path to FLAME model files")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--debug", action="store_true", help="Generate debug visualizations")
    parser.add_argument("--output", default="./output", help="Output directory")

    args = parser.parse_args()

    result = extract_lam_geometry(
        image_path=args.image,
        lam_repo_path=args.lam_repo,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        human_model_path=args.human_model_path,
        device=args.device,
        debug=args.debug,
        output_dir=args.output,
    )

    if result is None:
        print("\n[FAILED] Geometry extraction failed. See errors above.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] Geometry extracted successfully.")
        print("Next steps:")
        print("  1. Verify the 3D scatter plot matches a face shape")
        print("  2. Check the projection overlay on the image")
        print("  3. Confirm FLAME parameters were extracted")
        print("  4. Then proceed to webcam integration")


if __name__ == "__main__":
    main()
