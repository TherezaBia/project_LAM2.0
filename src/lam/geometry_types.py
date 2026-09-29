"""
PersonalizedFaceGeometry — geometry_types.py

Data class that holds the personalized facial geometry extracted from LAM.
This class represents the 3D structure of a specific person's face,
separate from any rendering or appearance information.

This is the bridge between LAM (geometry source) and the webcam pipeline
(appearance source).
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict
from pathlib import Path


@dataclass
class PersonalizedFaceGeometry:
    """
    Personalized 3D facial geometry from LAM.

    This class contains ONLY geometric information — no appearance data
    (no Gaussian colors, spherical harmonics, or rendered pixels).

    Attributes
    ----------
    canonical_xyz : np.ndarray [N, 3]
        Canonical 3D positions of all Gaussian/mesh points in FLAME space.
        These include the LAM-predicted offsets already applied to FLAME template.

    offsets : np.ndarray [N, 3]
        The delta offsets predicted by LAM on top of the FLAME template.
        canonical_xyz = flame_template_vertices + offsets

    query_points : np.ndarray [N, 3]
        The FLAME template vertices (before LAM offsets).
        Used as the "neutral" reference for deformation.

    flame_shape : np.ndarray [shape_dim]
        FLAME shape parameters (betas) — encode identity.

    faces : np.ndarray [F, 3] or None
        Face indices from the FLAME mesh topology.

    lbs_weights : np.ndarray [N, J] or None
        Linear Blend Skinning weights for each vertex.
        J is the number of FLAME joints.

    metadata : dict
        Additional metadata (device used, inference time, etc.)
    """

    canonical_xyz: np.ndarray
    offsets: np.ndarray
    query_points: Optional[np.ndarray] = None
    flame_shape: Optional[np.ndarray] = None
    faces: Optional[np.ndarray] = None
    lbs_weights: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def num_points(self) -> int:
        return self.canonical_xyz.shape[0]

    @property
    def xyz_range(self) -> Dict:
        return {
            "min": self.canonical_xyz.min(axis=0),
            "max": self.canonical_xyz.max(axis=0),
            "center": self.canonical_xyz.mean(axis=0),
        }

    @classmethod
    def from_npz(cls, path: str) -> "PersonalizedFaceGeometry":
        """Load from an .npz file produced by extract_geometry.py."""
        data = np.load(path, allow_pickle=True)

        return cls(
            canonical_xyz=data["xyz"] if "xyz" in data else None,
            offsets=data["offset"] if "offset" in data else None,
            query_points=data["query_points"] if "query_points" in data else None,
            flame_shape=data["flame_betas"] if "flame_betas" in data else None,
            faces=data["faces"] if "faces" in data else None,
            lbs_weights=data["lbs_weights"] if "lbs_weights" in data else None,
            metadata={
                "source_file": str(path),
                "keys": list(data.keys()),
            },
        )

    @classmethod
    def from_lam(cls, image_path: str, **kwargs) -> "PersonalizedFaceGeometry":
        """
        Create a PersonalizedFaceGeometry directly from an image using LAM.

        Parameters
        ----------
        image_path : str
            Path to a facial photograph.
        **kwargs
            Additional arguments passed to extract_lam_geometry().

        Returns
        -------
        PersonalizedFaceGeometry
        """
        from .geometry_bridge import extract_lam_geometry

        result = extract_lam_geometry(image_path, **kwargs)
        if result is None:
            raise RuntimeError("LAM geometry extraction failed")

        return cls(
            canonical_xyz=result.get("xyz"),
            offsets=result.get("offset"),
            query_points=result.get("query_points"),
            flame_shape=result.get("flame_betas"),
            faces=result.get("faces"),
            lbs_weights=result.get("lbs_weights"),
            metadata={
                "source_image": image_path,
                "flame_params": {
                    k: v.tolist() if isinstance(v, np.ndarray) and v.size < 100 else f"shape={v.shape}"
                    for k, v in result.items()
                    if k.startswith("flame_")
                },
            },
        )

    def deform(self, expression: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """
        Apply FLAME expression and pose deformation to the canonical geometry.

        This is a placeholder that will be fully implemented when the
        FLAME animation_forward integration is complete.

        Parameters
        ----------
        expression : np.ndarray
            FLAME expression parameters.
        pose : np.ndarray
            FLAME pose parameters (jaw, neck, eyes, rotation).

        Returns
        -------
        np.ndarray [N, 3]
            Deformed 3D positions.
        """
        # TODO: Integrate with FlameHeadSubdivided.animation_forward()
        # For now, return canonical positions
        raise NotImplementedError(
            "Deformation will be implemented in the tracking integration phase. "
            "This requires the FLAME model's animation_forward() function."
        )

    def save(self, path: str):
        """Save geometry to .npz file."""
        data = {"xyz": self.canonical_xyz, "offset": self.offsets}
        if self.query_points is not None:
            data["query_points"] = self.query_points
        if self.flame_shape is not None:
            data["flame_betas"] = self.flame_shape
        if self.faces is not None:
            data["faces"] = self.faces
        if self.lbs_weights is not None:
            data["lbs_weights"] = self.lbs_weights
        np.savez_compressed(path, **data)

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"PersonalizedFaceGeometry",
            f"  Points:       {self.num_points}",
            f"  XYZ shape:    {self.canonical_xyz.shape}",
            f"  Offset shape: {self.offsets.shape if self.offsets is not None else 'N/A'}",
            f"  XYZ range:    {self.xyz_range}",
            f"  Faces:        {self.faces.shape if self.faces is not None else 'N/A'}",
            f"  LBS weights:  {self.lbs_weights.shape if self.lbs_weights is not None else 'N/A'}",
            f"  FLAME shape:  {self.flame_shape.shape if self.flame_shape is not None else 'N/A'}",
        ]
        return "\n".join(lines)
