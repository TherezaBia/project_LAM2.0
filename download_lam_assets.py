"""
Download LAM model assets from HuggingFace.

Downloads:
  1. LAM-assets (FLAME model, tracking models, etc.) → model_zoo/
  2. LAM-20K checkpoint (model.safetensors)           → model_zoo/lam_models/
"""
import os
import sys
import tarfile
from pathlib import Path

def download_lam_assets(base_dir="."):
    """Download and extract LAM assets from HuggingFace."""
    from huggingface_hub import hf_hub_download

    base = Path(base_dir)
    lam_repo = base / "lam_repo"
    
    # ── 1. Download LAM human model (FLAME assets) ──
    human_model_tar = "LAM_human_model.tar"
    dest = lam_repo / human_model_tar
    
    flame_check = lam_repo / "model_zoo" / "human_parametric_models" / "flame_assets" / "flame" / "flame2023.pkl"
    if flame_check.exists():
        print(f"[skip] FLAME model already exists: {flame_check}")
    else:
        print(f"[1/3] Downloading {human_model_tar}...")
        try:
            path = hf_hub_download(
                repo_id="3DAIGC/LAM-assets",
                filename=human_model_tar,
                local_dir=str(lam_repo),
                repo_type="model",
            )
            print(f"       Downloaded to: {path}")
            
            # Extract
            print(f"       Extracting {human_model_tar}...")
            with tarfile.open(path, "r") as tar:
                tar.extractall(path=str(lam_repo))
            print(f"       Extracted ✓")
            
            # Clean up tar
            if os.path.exists(path):
                os.remove(path)
                
        except Exception as e:
            print(f"[ERROR] Failed to download {human_model_tar}: {e}")
            print(f"[ERROR] You may need to accept the license at https://huggingface.co/3DAIGC/LAM-assets")
    
    # ── 2. Download LAM assets (tracking models, etc.) ──
    assets_tar = "LAM_assets.tar"
    tracking_check = lam_repo / "model_zoo" / "flame_tracking_models"
    if tracking_check.exists():
        print(f"[skip] Tracking models already exist: {tracking_check}")
    else:
        print(f"[2/3] Downloading {assets_tar}...")
        try:
            path = hf_hub_download(
                repo_id="3DAIGC/LAM-assets",
                filename=assets_tar,
                local_dir=str(lam_repo),
                repo_type="model",
            )
            print(f"       Downloaded to: {path}")
            
            print(f"       Extracting {assets_tar}...")
            with tarfile.open(path, "r") as tar:
                tar.extractall(path=str(lam_repo))
            print(f"       Extracted ✓")
            
            if os.path.exists(path):
                os.remove(path)
                
        except Exception as e:
            print(f"[ERROR] Failed to download {assets_tar}: {e}")
    
    # ── 3. Download LAM-20K checkpoint ──
    ckpt_dir = lam_repo / "model_zoo" / "lam_models"
    ckpt_file = ckpt_dir / "model.safetensors"
    if ckpt_file.exists():
        print(f"[skip] LAM-20K checkpoint already exists: {ckpt_file}")
    else:
        print(f"[3/3] Downloading LAM-20K checkpoint...")
        os.makedirs(ckpt_dir, exist_ok=True)
        try:
            path = hf_hub_download(
                repo_id="3DAIGC/LAM-20K",
                filename="model.safetensors",
                local_dir=str(ckpt_dir),
                repo_type="model",
            )
            print(f"       Downloaded to: {path}")
        except Exception as e:
            print(f"[ERROR] Failed to download checkpoint: {e}")
            print(f"[ERROR] You may need to accept the license at https://huggingface.co/3DAIGC/LAM-20K")
    
    # ── Verify ──
    print(f"\n{'='*50}")
    print("Verification:")
    checks = [
        ("FLAME model", flame_check),
        ("Tracking models", tracking_check),
        ("LAM-20K checkpoint", ckpt_file),
    ]
    all_ok = True
    for name, path in checks:
        exists = path.exists()
        status = "✓" if exists else "✗ MISSING"
        print(f"  [{status}] {name}: {path}")
        if not exists:
            all_ok = False
    
    if all_ok:
        print("\nAll assets downloaded successfully!")
    else:
        print("\nSome assets are missing. Check errors above.")
    print(f"{'='*50}")
    
    return all_ok


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "."
    download_lam_assets(base)
