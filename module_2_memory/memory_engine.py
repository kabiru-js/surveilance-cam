import time
import uuid
from typing import Dict, Any, Optional
import numpy as np

from reid_extractor import VisualFingerprintExtractor
from vector_store import VectorStore
from camera_graph import CameraGraph

class SpatialTemporalMemoryEngine:
    """
    Module 2 Orchestrator — Spatial-Temporal Memory Engine.
    Fuses visual appearance fingerprints, vector similarity search,
    and spatial-temporal graph topology verification to link entities across cameras.
    """
    def __init__(
        self,
        db_path: str = "memory_vectors.db",
        reid_onnx_path: Optional[str] = None,
        min_similarity: float = 0.75
    ):
        self.extractor = VisualFingerprintExtractor(onnx_model_path=reid_onnx_path)
        self.vector_store = VectorStore(db_path=db_path)
        self.camera_graph = CameraGraph()
        self.min_similarity = min_similarity

    def process_observation(
        self,
        camera_id: str,
        timestamp: float,
        track_id: int,
        label: str,
        color: Optional[str] = None,
        plate: Optional[str] = None,
        crop_bgr: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Processes an incoming detection observation from Module 1:
        1. Extracts visual feature vector.
        2. Queries VectorStore for visually similar observations.
        3. Validates candidate transitions against CameraGraph kinematic bounds.
        4. Links to an existing global_entity_id or mints a new one.
        5. Persists the observation in the vector store.
        """
        # 1. Feature Extraction
        vector = self.extractor.extract(crop_bgr)

        # 2. Vector Similarity Search
        candidates = self.vector_store.search_similar(
            query_vector=vector,
            label=label,
            top_k=5,
            min_similarity=self.min_similarity
        )

        matched_candidate = None
        match_reason = "No visual match above threshold"

        # 3. Spatial-Temporal Plausibility Filtering
        for cand in candidates:
            cand_cam = cand["camera_id"]
            cand_ts = cand["timestamp"]
            cand_id = cand["global_entity_id"]

            # Vehicle license plate check
            plate_match = False
            if plate and cand.get("plate"):
                if plate.strip().upper() == cand["plate"].strip().upper():
                    plate_match = True

            # Color consistency penalty
            color_penalty = 0.0
            if color and cand.get("color"):
                c1, c2 = color.strip().lower(), cand["color"].strip().lower()
                if c1 != "unknown" and c2 != "unknown" and c1 != c2:
                    color_penalty = 0.25

            # Spatial-temporal transition plausibility
            is_valid, reason, topo_penalty = self.camera_graph.is_transition_plausible(
                cam_a=cand_cam,
                time_a=cand_ts,
                cam_b=camera_id,
                time_b=timestamp,
                label=label
            )

            if is_valid:
                total_penalty = min(1.0, topo_penalty + color_penalty)
                adj_similarity = cand["similarity"] * (1.0 - total_penalty)

                if adj_similarity >= self.min_similarity or plate_match:
                    matched_candidate = cand
                    match_reason = reason
                    break
                else:
                    match_reason = f"Rejected candidate {cand_id}: similarity after penalties ({adj_similarity:.2f}) below threshold"
            else:
                # Spatial-temporal violation (e.g., teleportation)
                match_reason = f"Rejected visual match with {cand_id}: {reason}"

        # 4. Entity Resolution (Link or Mint)
        if matched_candidate is not None:
            action = "LINKED"
            global_entity_id = matched_candidate["global_entity_id"]
            matched_info = {
                "matched_entity_id": global_entity_id,
                "matched_camera": matched_candidate["camera_id"],
                "matched_timestamp": matched_candidate["timestamp"],
                "visual_similarity": matched_candidate["similarity"],
                "spatial_temporal_status": match_reason
            }
        else:
            action = "NEW"
            short_lbl = label.lower()[:3]
            short_uuid = uuid.uuid4().hex[:6]
            global_entity_id = f"ent_{short_lbl}_{int(timestamp)}_{short_uuid}"
            matched_info = {
                "spatial_temporal_status": match_reason
            }

        # 5. Store Observation in Persistent Memory
        embedding_id = self.vector_store.insert_embedding(
            global_entity_id=global_entity_id,
            camera_id=camera_id,
            track_id=track_id,
            label=label,
            color=color,
            plate=plate,
            timestamp=timestamp,
            vector=vector
        )

        result = {
            "action": action,
            "global_entity_id": global_entity_id,
            "embedding_id": embedding_id,
            "camera_id": camera_id,
            "track_id": track_id,
            "label": label,
            "color": color,
            "plate": plate,
            "timestamp": timestamp,
            "matched_info": matched_info
        }

        return result

    def close(self):
        self.vector_store.close()
