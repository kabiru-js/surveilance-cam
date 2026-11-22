import sqlite3
import time
import uuid
from typing import List, Dict, Optional
import numpy as np

class VectorStore:
    """
    Zero-dependency local persistent vector storage backed by SQLite and NumPy.
    Performs fast in-memory matrix cosine similarity searches over L2-normalized float32 embeddings.
    Designed to allow seamless transition to PostgreSQL + pgvector.
    """
    def __init__(self, db_path: str = "memory_vectors.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    embedding_id TEXT PRIMARY KEY,
                    global_entity_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    color TEXT,
                    plate TEXT,
                    timestamp REAL NOT NULL,
                    vector BLOB NOT NULL
                );
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_label ON embeddings(label);")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_entity ON embeddings(global_entity_id);")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON embeddings(timestamp);")

    def insert_embedding(
        self,
        global_entity_id: str,
        camera_id: str,
        track_id: int,
        label: str,
        vector: np.ndarray,
        color: Optional[str] = None,
        plate: Optional[str] = None,
        timestamp: Optional[float] = None,
        embedding_id: Optional[str] = None
    ) -> str:
        """
        Serializes and stores an L2-normalized feature vector with its metadata.
        Returns the embedding_id.
        """
        if embedding_id is None:
            embedding_id = f"emb_{uuid.uuid4().hex[:12]}"
        if timestamp is None:
            timestamp = time.time()

        # Ensure float32 1D array
        vec_bytes = np.asarray(vector, dtype=np.float32).tobytes()

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO embeddings (
                    embedding_id, global_entity_id, camera_id, track_id,
                    label, color, plate, timestamp, vector
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    embedding_id,
                    global_entity_id,
                    camera_id,
                    int(track_id),
                    label.lower(),
                    color,
                    plate,
                    float(timestamp),
                    vec_bytes
                )
            )
        return embedding_id

    def search_similar(
        self,
        query_vector: np.ndarray,
        label: str,
        top_k: int = 5,
        min_similarity: float = 0.75,
        exclude_camera_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Searches stored embeddings filtered by object label using vectorized cosine similarity.
        Since vectors are L2-normalized, cosine similarity reduces to matrix-vector dot product.
        """
        cursor = self.conn.cursor()
        
        query = """
            SELECT embedding_id, global_entity_id, camera_id, track_id,
                   label, color, plate, timestamp, vector
            FROM embeddings
            WHERE label = ?
        """
        params = [label.lower()]

        if exclude_camera_id:
            query += " AND camera_id != ?"
            params.append(exclude_camera_id)

        query += " ORDER BY timestamp DESC"
        cursor.execute(query, params)
        rows = cursor.fetchall()

        if not rows:
            return []

        # Deserialize all vectors for fast batch matrix multiplication
        meta_list = []
        vectors = []

        q_vec = np.asarray(query_vector, dtype=np.float32).flatten()
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 1e-6:
            q_vec = q_vec / q_norm

        dim = len(q_vec)

        for row in rows:
            emb_id, g_id, cam_id, trk_id, lbl, col, plt, ts, blob = row
            vec = np.frombuffer(blob, dtype=np.float32)
            if len(vec) == dim:
                vectors.append(vec)
                meta_list.append({
                    "embedding_id": emb_id,
                    "global_entity_id": g_id,
                    "camera_id": cam_id,
                    "track_id": trk_id,
                    "label": lbl,
                    "color": col,
                    "plate": plt,
                    "timestamp": ts
                })

        if not vectors:
            return []

        matrix = np.vstack(vectors)  # Shape (N, 128)
        similarities = matrix @ q_vec  # Dot product = cosine similarity

        # Filter by threshold and rank
        valid_indices = np.where(similarities >= min_similarity)[0]
        if len(valid_indices) == 0:
            return []

        # Sort descending by similarity
        sorted_indices = valid_indices[np.argsort(-similarities[valid_indices])][:top_k]

        results = []
        for idx in sorted_indices:
            match = dict(meta_list[idx])
            match["similarity"] = round(float(similarities[idx]), 4)
            results.append(match)

        return results

    def get_entity_history(self, global_entity_id: str) -> List[Dict]:
        """
        Retrieves complete chronological observation trail for an entity across cameras.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT embedding_id, global_entity_id, camera_id, track_id,
                   label, color, plate, timestamp
            FROM embeddings
            WHERE global_entity_id = ?
            ORDER BY timestamp ASC
            """,
            (global_entity_id,)
        )
        rows = cursor.fetchall()
        history = []
        for r in rows:
            history.append({
                "embedding_id": r[0],
                "global_entity_id": r[1],
                "camera_id": r[2],
                "track_id": r[3],
                "label": r[4],
                "color": r[5],
                "plate": r[6],
                "timestamp": r[7]
            })
        return history

    def count(self) -> int:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        return cursor.fetchone()[0]

    def close(self):
        if self.conn:
            self.conn.close()
