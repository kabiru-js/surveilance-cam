from typing import Tuple, Optional
import networkx as nx

class CameraGraph:
    """
    Spatial-Temporal Camera Topology Graph powered by NetworkX.
    Models physical layout, distances, and travel plausibility across cameras.
    """
    PEDESTRIAN_MAX_SPEED = 5.0   # m/s (~18 km/h sprint limit)
    PEDESTRIAN_NORMAL_SPEED = 1.4 # m/s (~5 km/h normal walk)
    VEHICLE_MAX_SPEED = 18.0     # m/s (~65 km/h max speed on estate roads)
    VEHICLE_NORMAL_SPEED = 8.0   # m/s (~29 km/h typical estate drive)

    VEHICLE_LABELS = {"car", "bus", "truck", "motorcycle"}

    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_default_estate_topology()

    def _build_default_estate_topology(self):
        """
        Builds default realistic estate camera network:
        - CAM_GATE_A: Main entrance gate
        - CAM_DRIVEWAY: Central access road
        - CAM_PARKING_LOT: Main parking structure
        - CAM_LOBBY: Building reception & pedestrian entrance
        """
        nodes = [
            ("CAM_GATE_A", {"zone": "Perimeter", "type": "entry_exit"}),
            ("CAM_DRIVEWAY", {"zone": "Transit", "type": "roadway"}),
            ("CAM_PARKING_LOT", {"zone": "Vehicular", "type": "parking"}),
            ("CAM_LOBBY", {"zone": "Interior_Pedestrian", "type": "entrance"})
        ]
        self.graph.add_nodes_from(nodes)

        # Bi-directional connectivity with physical distance in meters
        edges = [
            # Gate <-> Driveway: 120m
            ("CAM_GATE_A", "CAM_DRIVEWAY", 120.0),
            ("CAM_DRIVEWAY", "CAM_GATE_A", 120.0),

            # Driveway <-> Parking Lot: 180m (Gate -> Driveway -> Parking = 300m)
            ("CAM_DRIVEWAY", "CAM_PARKING_LOT", 180.0),
            ("CAM_PARKING_LOT", "CAM_DRIVEWAY", 180.0),

            # Gate <-> Lobby: 50m direct pedestrian walkway
            ("CAM_GATE_A", "CAM_LOBBY", 50.0),
            ("CAM_LOBBY", "CAM_GATE_A", 50.0),

            # Driveway <-> Lobby: 100m pathway
            ("CAM_DRIVEWAY", "CAM_LOBBY", 100.0),
            ("CAM_LOBBY", "CAM_DRIVEWAY", 100.0),

            # Parking Lot <-> Lobby: 250m walkway (Gate -> Lobby -> Parking = 300m)
            ("CAM_PARKING_LOT", "CAM_LOBBY", 250.0),
            ("CAM_LOBBY", "CAM_PARKING_LOT", 250.0)
        ]

        for u, v, dist in edges:
            self.graph.add_edge(u, v, distance_meters=dist)

    def add_camera(self, camera_id: str, zone: str = "General", camera_type: str = "standard"):
        self.graph.add_node(camera_id, zone=zone, type=camera_type)

    def add_edge(self, cam_a: str, cam_b: str, distance_meters: float, bidirectional: bool = True):
        self.graph.add_edge(cam_a, cam_b, distance_meters=float(distance_meters))
        if bidirectional:
            self.graph.add_edge(cam_b, cam_a, distance_meters=float(distance_meters))

    def get_shortest_distance(self, cam_a: str, cam_b: str) -> Optional[float]:
        if cam_a == cam_b:
            return 0.0
        try:
            return nx.shortest_path_length(self.graph, source=cam_a, target=cam_b, weight="distance_meters")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def is_transition_plausible(
        self,
        cam_a: str,
        time_a: float,
        cam_b: str,
        time_b: float,
        label: str = "person"
    ) -> Tuple[bool, str, float]:
        """
        Validates whether transition from (cam_a, time_a) to (cam_b, time_b)
        is physically plausible according to the spatial graph and kinematic bounds.
        
        Returns:
            (is_valid: bool, reason: str, confidence_penalty: float)
        """
        delta_t = time_b - time_a

        # 1. Chronological consistency
        if delta_t < 0:
            return False, f"Negative time delta ({delta_t:.2f}s): backward time transition impossible", 1.0

        # 2. Same-camera observation
        if cam_a == cam_b:
            if delta_t <= 30.0:
                return True, f"Continuous observation on same camera ({delta_t:.1f}s)", 0.0
            else:
                penalty = min(0.3, delta_t / 600.0)
                return True, f"Reappearance on same camera after {delta_t:.1f}s", penalty

        # 3. Node existence check
        if cam_a not in self.graph or cam_b not in self.graph:
            return False, f"Camera topology missing node: {cam_a} or {cam_b}", 1.0

        # 4. Shortest path distance
        dist = self.get_shortest_distance(cam_a, cam_b)
        if dist is None:
            return False, f"No topological path between {cam_a} and {cam_b}", 1.0

        # 5. Kinematic limits
        is_veh = label.lower() in self.VEHICLE_LABELS
        max_speed = self.VEHICLE_MAX_SPEED if is_veh else self.PEDESTRIAN_MAX_SPEED
        normal_speed = self.VEHICLE_NORMAL_SPEED if is_veh else self.PEDESTRIAN_NORMAL_SPEED

        implied_speed = dist / max(delta_t, 0.001)

        # Teleportation check: speed exceeded max physical capability
        if implied_speed > max_speed:
            return (
                False,
                f"Teleportation violation: speed {implied_speed:.1f} m/s exceeds max threshold {max_speed} m/s "
                f"({dist:.1f}m in {delta_t:.1f}s)",
                1.0
            )

        # Extended delay check: entity lingered
        expected_time = dist / normal_speed
        max_transit_time = expected_time * 8.0 + 300.0  # 5 min grace window

        if delta_t > max_transit_time:
            penalty = min(0.4, (delta_t - max_transit_time) / 600.0)
            return True, f"Plausible transition with lingering ({dist:.1f}m in {delta_t:.1f}s)", penalty

        return True, f"Plausible transition: {dist:.1f}m in {delta_t:.1f}s ({implied_speed:.2f} m/s)", 0.0
