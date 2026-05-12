"""EnvironmentManager agent for providing environment information and preferences."""

import requests
import threading
from typing import Any, Optional
from models import (
    ArtifactInfo,
    ArtifactState,
    ActionAffordance,
    PropertyAffordance,
    Preference,
    TimeInterval,
)
from agent_protocol import AgentMailbox, MessageType, get_message_broker


class EnvironmentManagerAgent:
    """
    Agent responsible for providing environment information and active user preferences.

    This agent implements four deterministic methods for accessing simulator state
    and preference constraints. All methods use pure HTTP queries — no LLM calls.

    Students should implement:
      - get_rooms()
      - get_artifacts_in_room()
      - get_artifact_affordances()
      - get_artifact_state()
      - get_active_preferences()
    """

    def __init__(
        self,
        simulator_base_url: str,
        preferences: list[Preference],
        timeout: int = 30,
        verbose: bool = False,
    ):
        """
        Initialize the EnvironmentManager.

        Args:
            simulator_base_url: Base URL of the SmartHome simulator (e.g., http://localhost:8080)
            preferences: List of user preference constraints
            timeout: HTTP request timeout in seconds
            verbose: Enable verbose logging of agent communication
        """
        self.simulator_url = simulator_base_url.rstrip("/")
        self.preferences = preferences
        self.timeout = timeout
        self.verbose = verbose
        self.mailbox = AgentMailbox("EnvironmentManager")
        get_message_broker().register_agent("EnvironmentManager", self.mailbox)
        self._running = False
        self._thread = None

    def start(self):
        """Start the agent message handler thread."""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        """Stop the agent message handler thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self):
        """Main message handling loop (runs in thread)."""
        while self._running:
            msg = self.mailbox.receive(timeout=1.0)
            if msg is None:
                continue

            if msg.message_type == MessageType.SHUTDOWN:
                break

            if msg.message_type == MessageType.REQUEST:
                self._handle_request(msg)

    def _handle_request(self, msg):
        """Handle incoming request message."""
        method = msg.method
        args = msg.args
        request_id = msg.request_id
        sender = msg.sender

        if self.verbose:
            print(f"[EnvironmentManager] ← Request from {sender}: {method}({args})")

        try:
            if method == "get_rooms":
                result = self.get_rooms(**args)
            elif method == "get_artifacts_in_room":
                result = self.get_artifacts_in_room(**args)
            elif method == "get_artifact_affordances":
                result = self.get_artifact_affordances(**args)
            elif method == "get_artifact_state":
                result = self.get_artifact_state(**args)
            elif method == "get_active_preferences":
                result = self.get_active_preferences(**args)
            elif method == "read_property":
                result = self.read_property(**args)
            else:
                raise ValueError(f"Unknown method: {method}")

            if self.verbose:
                print(f"[EnvironmentManager] → Response to {sender}: {method} = {result}")
            get_message_broker().route_response(request_id, result)

        except Exception as e:
            if self.verbose:
                print(f"[EnvironmentManager] ✗ Error handling {method}: {e}")
            get_message_broker().route_response(request_id, None, error=str(e))

    def read_property(self, property_uri: str) -> Any:
        """Read a property value from the simulator."""
        try:
            response = requests.get(property_uri, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def get_rooms(self, home_id: str) -> list[str]:
        """
        Return the list of room names (sub-workspace names) for a given home.

        This method should:
          1. Construct the home workspace URI
          2. Fetch the RDF/Turtle description
          3. Parse the HMAS workspace hierarchy to find sub-workspaces (rooms)
          4. Return room names as strings

        Args:
            home_id: Home identifier (e.g., "home71")

        Returns:
            List of room names (e.g., ["master_bedroom", "living_room", "kitchen"])

        Raises:
            NotImplementedError: Students must implement this method
        """
        raise NotImplementedError

    def get_artifacts_in_room(self, home_id: str, room: str) -> list[str]:
        """
        Return the list of artifact URIs present in a given room.

        This method queries the simulator for the room's RDF description and
        parses it to find all artifacts (devices) contained in that room.

        Args:
            home_id: Home identifier (e.g., "home71")
            room: Room name (e.g., "guest_bedroom")

        Returns:
            List of artifact URIs or names
        """
        try:
            # Fetch room workspace RDF
            room_uri = f"{self.simulator_url}/workspaces/{home_id}/{room}"
            response = requests.get(room_uri, timeout=self.timeout)
            response.raise_for_status()
            rdf_content = response.text

            # Simple regex-based parsing for artifact URIs
            # Pattern: hmas:contains <http://...#artifact>
            import re
            pattern = r'hmas:contains\s+<(http[^>]+/artifacts/[^>]+)>'
            artifacts = re.findall(pattern, rdf_content)

            if self.verbose:
                print(f"[EnvironmentManager] Found {len(artifacts)} artifacts in {home_id}/{room}")

            return artifacts
        except Exception as e:
            if self.verbose:
                print(f"[EnvironmentManager] Error in get_artifacts_in_room: {e}")
            return []

    def get_artifact_affordances(self, artifact_uri: str) -> ArtifactInfo:
        """
        Return the affordances (actions and properties) for a single artifact.

        This method should:
          1. Fetch the artifact's Thing Description (TD) from the simulator
          2. Parse the WoT TD to extract:
             - Action affordances (with input parameter schemas)
             - Property affordances (with URIs for reading/writing)
          3. Return an ArtifactInfo object with these affordances

        Args:
            artifact_uri: Full URI of the artifact

        Returns:
            ArtifactInfo with actions and properties populated

        Raises:
            NotImplementedError: Students must implement this method
        """
        raise NotImplementedError

    def get_artifact_state(
        self,
        artifact_uri: str,
        property_name: Optional[str] = None,
    ) -> ArtifactState:
        """
        Fetch current property values for an artifact from the simulator.

        This method should:
          1. If property_name is given, fetch only that property's value
          2. If property_name is None, fetch all properties for the artifact
          3. Make HTTP GET requests to the property URIs
          4. Return an ArtifactState with the properties dict populated

        Args:
            artifact_uri: Full URI of the artifact
            property_name: Optional specific property name; if None, fetch all

        Returns:
            ArtifactState with current property values

        Raises:
            NotImplementedError: Students must implement this method
        """
        raise NotImplementedError

    def get_active_preferences(self, issued_at: str) -> list[Preference]:
        """
        Return preferences whose dislike interval contains the given time.

        This method should:
          1. Parse the issued_at string (format "HH:MM")
          2. For each preference, check if issued_at falls within [start, end)
          3. Return only the active preferences

        Args:
            issued_at: Time string in "HH:MM" format

        Returns:
            List of active Preference objects

        Raises:
            NotImplementedError: Students must implement this method
        """
        raise NotImplementedError
