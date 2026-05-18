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
import rdflib
from rdflib import Graph, Namespace, URIRef, RDF
from models import ArtifactInfo, ArtifactState, ActionAffordance, PropertyAffordance, Preference


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
        url = f"{self.simulator_url}/workspaces/{home_id}"
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()

        g = rdflib.Graph()
        g.parse(data=response.text, format="turtle")
        hmas = rdflib.Namespace("https://purl.org/hmas/")

        rooms = []
        # Find all entities contained in the home workspace
        for _, _, obj in g.triples((None, hmas.contains, None)):
            uri_str = str(obj)
            # Rooms are sub-workspaces, so they usually end with #workspace
            if "#workspace" in uri_str:
                # Extract the room name from the URI path
                room_name = uri_str.split('/')[-1].split('#')[0]
                rooms.append(room_name)
                
        if self.verbose:
            print(f"[EnvironmentManager] Found {len(rooms)} rooms in {home_id}")
            
        return rooms

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
            url = f"{self.simulator_url}/workspaces/{home_id}/{room}"
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            g = rdflib.Graph()
            g.parse(data=response.text, format="turtle")
            hmas = rdflib.Namespace("https://purl.org/hmas/")
            
            artifacts = []
            for _, _, obj in g.triples((None, hmas.contains, None)):
                uri_str = str(obj)
                if "#artifact" in uri_str:
                    artifacts.append(uri_str)
                    
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
        url = artifact_uri.split("#")[0]
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()

        g = rdflib.Graph()
        g.parse(data=response.text, format="turtle")

        td = rdflib.Namespace("https://www.w3.org/2019/wot/td#")
        hctl = rdflib.Namespace("https://www.w3.org/2019/wot/hypermedia#")
        jsonschema = rdflib.Namespace("https://www.w3.org/2019/wot/json-schema#")

        parts = url.rstrip("/").split("/")
        artifact_name = parts[-1]
        room_name = parts[-3] if len(parts) >= 3 else "unknown"

        # Determine the abstract device type
        subject_uri = rdflib.URIRef(artifact_uri)
        
        # Determine the abstract device type specifically for this artifact
        device_type = "unknown"
        for _, _, obj in g.triples((subject_uri, rdflib.RDF.type, None)):
            if str(obj).startswith("http://example.org/"):
                device_type = str(obj).replace("http://example.org/", "")
                # Some TDs might have multiple example.org types, make sure we don't grab the room
                if device_type not in ["GuestBedroom", "MasterBedroom", "LivingRoom", "Bathroom", "StudyRoom", "Garage", "Corridor", "Balcony"]:
                    break

        info = ArtifactInfo(
            name=artifact_name,
            room=room_name,
            artifact_uri=artifact_uri,
            device_type=device_type,
            actions=[],
            properties=[]
        )

        subject_uri = rdflib.URIRef(artifact_uri)

        # Parse Property Affordances
        for prop_aff in g.objects(subject_uri, td.hasPropertyAffordance):
            name = str(next(g.objects(prop_aff, td.name), ""))
            target = ""
            for form in g.objects(prop_aff, td.hasForm):
                target = str(next(g.objects(form, hctl.hasTarget), ""))
                break
            if name and target:
                info.properties.append(PropertyAffordance(name=name, uri=target))

        # Parse Action Affordances
        for act_aff in g.objects(subject_uri, td.hasActionAffordance):
            name = str(next(g.objects(act_aff, td.name), ""))
            target = ""
            for form in g.objects(act_aff, td.hasForm):
                target = str(next(g.objects(form, hctl.hasTarget), ""))
                break

            # Process input parameter schema requirements
            input_schema_dict = {}
            for input_schema in g.objects(act_aff, td.hasInputSchema):
                for prop in g.objects(input_schema, jsonschema.properties):
                    param_name = str(next(g.objects(prop, jsonschema.propertyName), ""))
                    if param_name:
                        param_type = "unknown"
                        for t in g.objects(prop, rdflib.RDF.type):
                            if "Schema" in str(t):
                                param_type = str(t).split("#")[-1].replace("Schema", "").lower()
                        input_schema_dict[param_name] = {"type": param_type}

            if name and target:
                info.actions.append(ActionAffordance(name=name, uri=target, input_schema=input_schema_dict))

        return info

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
        info = self.get_artifact_affordances(artifact_uri)
        state = ArtifactState(artifact_uri=artifact_uri, properties={})

        for prop in info.properties:
            # Skip if we are targeting a specific property and this isn't it
            if property_name and prop.name != property_name:
                continue
                
            # Make the HTTP GET request using the provided helper
            result = self.read_property(prop.uri)
            
            # The simulator returns an object with a "value" key for ObjectSchemas, 
            # or primitive JSON data for others. Unwrap if needed.
            if isinstance(result, dict) and "value" in result:
                state.properties[prop.name] = result["value"]
            elif isinstance(result, dict) and "error" in result:
                state.properties[prop.name] = None # Or handle the exception appropriately
            else:
                state.properties[prop.name] = result

        return state

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
        active_prefs = []
        for pref in self.preferences:
            start_time = pref.dislike_interval.start
            end_time = pref.dislike_interval.end
            
            # Because timestamps are zero-padded "HH:MM", we can evaluate directly with string comparison.
            if start_time <= issued_at < end_time:
                active_prefs.append(pref)
                
        if self.verbose and active_prefs:
            print(f"[EnvironmentManager] Found {len(active_prefs)} active preferences at {issued_at}")
            
        return active_prefs
