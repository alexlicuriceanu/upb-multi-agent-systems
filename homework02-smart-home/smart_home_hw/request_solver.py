"""RequestSolver agent for handling smart home requests."""

import os
import re
import threading
import time
from models import Request, ActionOutput
from environment_manager import EnvironmentManagerAgent
from agent_protocol import AgentMailbox, MessageType, get_message_broker
import json
from llm_client import call_llm
import dataclasses


class RequestSolverAgent:
    """
    Agent responsible for interpreting and executing smart home requests.

    This agent communicates with EnvironmentManagerAgent to discover capabilities,
    apply user preferences, and generate action sequences.

    Students must implement the solve() method with three distinct strategies.
    """

    def __init__(
        self,
        env_manager: EnvironmentManagerAgent,
        llm_client,
    ):
        """
        Initialize the RequestSolver.

        Args:
            env_manager: EnvironmentManagerAgent instance for accessing environment info
            llm_client: LLM client for reasoning (from llm_client module)
        """
        self.env_manager = env_manager
        self.llm_client = llm_client

    def solve(self, request: Request) -> list[ActionOutput]:
        """
        Interpret a natural language request and return predicted action outputs.

        This method should:
          1. Parse the request (request.input and request.issued_at)
          2. Call env_manager.get_artifact_affordances() and get_artifact_state()
             to understand available capabilities and current state
          3. Call env_manager.get_active_preferences(request.issued_at) to check
             for user constraints
          4. Use the LLM to interpret the request and map to specific actions
          5. For each sub-goal:
             - If device is unavailable (not present in any room): return error_input
             - If a preference blocks this device+room at this time: return error_input
             - If feasible: return ActionOutput with affordance, params

        Contract (what you return):
          - For infeasible sub-goals (device absent OR preference active):
            Return ActionOutput(execution="error_input")
          - For feasible sub-goals:
            Return ActionOutput(execution="success", affordance=url, params=dict)
              where affordance is the action URI and params is a dict of action parameters.
              Do NOT populate the test field—the evaluation engine will do that.

        (The evaluation engine will then execute your predicted action at the simulator,
         fetch the property values from ground truth, and verify correctness.)

        Args:
            request: Request object

        Returns:
            List of ActionOutput objects (one per sub-goal in the request)

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError


class DummyRequestSolver(RequestSolverAgent):
    """Dummy solver for testing the evaluation harness and agent communication."""

    def __init__(self, env_manager: EnvironmentManagerAgent, llm_client, verbose: bool = False):
        """Initialize with agent mailbox for communication."""
        super().__init__(env_manager, llm_client)
        self.verbose = verbose
        self.mailbox = AgentMailbox("RequestSolver")
        get_message_broker().register_agent("RequestSolver", self.mailbox)

    def solve(self, request: Request) -> list[ActionOutput]:
        """
        Demonstrate agent communication by querying EnvironmentManager.

        This solver:
        1. Requests artifacts in guest_bedroom of home12
        2. Asks for artifact info on guestBedroomDehumidifiers
        3. Reads the mode property
        4. Returns the dummy action
        """
        home_id = "home12"
        room = "guest_bedroom"
        artifact_name = "guestBedroomDehumidifiers"

        if self.verbose:
            print(f"\n[DummyRequestSolver] Processing request: {request.id}")
            print(f"[DummyRequestSolver] Home: {home_id}, Room: {room}")

        # Step 1: Request artifacts in guest_bedroom
        if self.verbose:
            print(f"[DummyRequestSolver] → Requesting artifacts in {room}")
        try:
            artifacts = self.mailbox.request(
                "EnvironmentManager",
                "get_artifacts_in_room",
                {"home_id": home_id, "room": room},
                timeout=5.0
            )
            if self.verbose:
                print(f"[DummyRequestSolver] ← Received {len(artifacts) if isinstance(artifacts, list) else 1} artifact(s)")
        except Exception as e:
            if self.verbose:
                print(f"[DummyRequestSolver] ✗ Error getting artifacts: {e}")
            artifacts = []

        # Step 2: Request artifact affordances for guestBedroomDehumidifiers
        artifact_uri = f"http://localhost:8080/workspaces/{home_id}/{room}/artifacts/{artifact_name}"
        if self.verbose:
            print(f"[DummyRequestSolver] → Requesting affordances for {artifact_name}")
        try:
            affordances = self.mailbox.request(
                "EnvironmentManager",
                "get_artifact_affordances",
                {"artifact_uri": artifact_uri},
                timeout=5.0
            )
            if self.verbose:
                print(f"[DummyRequestSolver] ← Received affordance info")
        except Exception as e:
            if self.verbose:
                print(f"[DummyRequestSolver] ✗ Error getting affordances: {e}")
            affordances = None

        # Step 3: Request to read the mode property
        property_uri = f"{artifact_uri}/properties/mode"
        if self.verbose:
            print(f"[DummyRequestSolver] → Reading property: mode")
        try:
            property_value = self.mailbox.request(
                "EnvironmentManager",
                "read_property",
                {"property_uri": property_uri},
                timeout=5.0
            )
            if self.verbose:
                print(f"[DummyRequestSolver] ← Current mode value: {property_value}")
        except Exception as e:
            if self.verbose:
                print(f"[DummyRequestSolver] ✗ Error reading property: {e}")
            property_value = None

        # Step 4: Return the dummy action (regardless of what we received)
        if self.verbose:
            print(f"[DummyRequestSolver] → Returning hardcoded action")
        return [
            ActionOutput(
                execution="success",
                affordance="http://localhost:8080/workspaces/home12/guest_bedroom/artifacts/guestBedroomDehumidifiers/set_mode",
                params={"mode": "auto"}
            )
        ]


class FullContextSolver(RequestSolverAgent):
    """Strategy 1: Full Context"""

    def __init__(self, env_manager, llm_client, verbose: bool = False):
        super().__init__(env_manager, llm_client)
        self.verbose = verbose
        self.test_counter = 0
        
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = "logs" 
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_filename = f"{self.log_dir}/{self.__class__.__name__}_{current_time}.txt"
        
        if self.verbose:
            print(f"[{self.__class__.__name__}] All logs will be saved to: {self.log_filename}")

    def solve(self, request: Request) -> list[ActionOutput]:
        self.test_counter += 1  # Increment counter for each request
        home_id = request.id.split("_")[0]
        
        def log_message(msg: str):
            """Helper to print conditionally but write to file unconditionally."""
            if self.verbose:
                print(msg)
            with open(self.log_filename, "a", encoding="utf-8") as log_file:
                log_file.write(msg + "\n")

        # Extract and format the expected output (ground truth)
        expected_dicts = [dataclasses.asdict(o) for o in request.output]
        expected_json = json.dumps(expected_dicts, indent=2)

        # Log the starting banner with Test ID and Expected Output
        banner = (
            f"\n{'='*60}\n"
            f"[{self.test_counter}/36] Request ID: {request.id}\n"
            f"User Input: '{request.input}'\n"
            f"Time: {request.issued_at} | Home: {home_id}\n"
            f"\nEXPECTED OUTPUT (GROUND TRUTH)\n"
            f"{expected_json}\n"
            f"{'='*60}"
        )
        log_message(banner)

        # Gather all environment information
        rooms = self.env_manager.get_rooms(home_id)
        
        environment_state = {}
        for room in rooms:
            artifacts = self.env_manager.get_artifacts_in_room(home_id, room)
            room_data = []
            for artifact_uri in artifacts:
                affordances = self.env_manager.get_artifact_affordances(artifact_uri)
                state = self.env_manager.get_artifact_state(artifact_uri)
                
                device_info = {
                    "device_type": affordances.device_type,
                    "uri": artifact_uri,
                    "current_state": state.properties,
                    "actions": [{"name": a.name, "uri": a.uri, "schema": a.input_schema} for a in affordances.actions]
                }
                room_data.append(device_info)
            if room_data:
                environment_state[room] = room_data

        # Get active preferences
        active_prefs = self.env_manager.get_active_preferences(request.issued_at)
        prefs_data = [{"device": p.device_type, "room": p.room, "reason": p.reason} for p in active_prefs]

        # Construct the JSON-Only Prompt
        prompt = f"""
You are an expert Smart Home AI mapping user requests to precise actions.

User Request: "{request.input}"
Time of Request: {request.issued_at}

Active Constraints:
If a requested device and room match ANY of these constraints, the action is BLOCKED. 
{json.dumps(prefs_data, indent=2)}

Environment State:
This is the ONLY source of truth for devices in the house.
{json.dumps(environment_state, indent=2)}

Instructions:
Analyze the request and break it into sub-goals. 
You MUST output ONLY a valid JSON array. 

CRITICAL RULES:
1. STRICT DEVICE MATCHING: If the user asks for a "fan", they mean a standalone Fan. If they ask for "heating", they mean a standalone Heating device. An AirConditioner is NOT a Fan and NOT a Heating device. If the exact requested device type is missing, fail with "error_input".
2. STRICT PARAMETER MATCHING: You can only adjust a parameter if it explicitly exists in the device's schema. Do NOT substitute parameters (e.g., using "color" to adjust "brightness"). If the requested parameter is missing, fail with "error_input".
3. STATELESS MATH RULE: All calculations MUST be based on the INITIAL `current_state` provided. If there are multiple sequential sub-goals for the same device (e.g., "decrease by X%, then decrease by Y%"), treat each as an INDEPENDENT calculation against the original starting state.
4. PERCENTAGE MATH = ABSOLUTE POINTS: Treat "percent" or "%" as ABSOLUTE raw numbers. (e.g., "Decrease brightness by 23%" from 83 means 83 - 23 = 60).
5. STRICT VOCABULARY: The "execution" key MUST be exactly "success" or "error_input".

For each sub-goal, your "reasoning" key MUST follow this EXACT structure:
"Target: [Device]. Room contains: [List exact devices found in the JSON for this room]. Exists: [Yes/No]. Blocked: [Yes/No]. Math: [Original State +/- Change = Final]."

OUTPUT FORMAT MUST MATCH THIS EXACTLY:
[
  {{
    "reasoning": "Target: Curtain. Room contains: [Aromatherapy, Fan, Heating, Humidifier, Light]. Exists: No. Execution: error_input.",
    "execution": "error_input"
  }},
  {{
    "reasoning": "Target: Light. Room contains: [Light, Trash]. Exists: Yes. Blocked: No. Math: 83 - 23 = 60.",
    "execution": "success",
    "affordance": "action_URI",
    "params": {{ "actual_parameter_name_from_schema": 60 }}
  }}
]
"""

        # Call LLM
        raw_response = call_llm(self.llm_client, prompt)

        log_message(f"[FullContextSolver] LLM Raw Response:\n{raw_response}\n")

        # Extract and Parse JSON
        try:
            clean_text = raw_response.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            
            json_match = re.search(r'\[\s*\{.*\}\s*\]', clean_text.strip(), re.DOTALL)
            
            if not json_match:
                log_message("[FullContextSolver] Regex failed to find JSON array.")
                return [ActionOutput(execution="error_input")]
                
            clean_json = json_match.group(0)
            parsed_outputs = json.loads(clean_json)
            
            final_outputs = []
            for item in parsed_outputs:
                if item.get("execution") == "error_input":
                    final_outputs.append(ActionOutput(execution="error_input"))
                else:
                    final_outputs.append(ActionOutput(
                        execution="success",
                        affordance=item.get("affordance"),
                        params=item.get("params", {})
                    ))
            return final_outputs
            
        except json.JSONDecodeError as e:
            log_message(f"[FullContextSolver] JSON Parsing Error: {e}")
            return [ActionOutput(execution="error_input")]
        except Exception as e:
            log_message(f"[FullContextSolver] Unexpected Error: {e}")
            return [ActionOutput(execution="error_input")]

class SequentialSolver(RequestSolverAgent):
    """Strategy 2: Sequential Exploration - not yet implemented."""

    def solve(self, request: Request) -> list[ActionOutput]:
        raise NotImplementedError("Strategy 2: Sequential Exploration not yet implemented")


class SemanticSolver(RequestSolverAgent):
    """Strategy 3: Semantic Classification - not yet implemented."""

    def solve(self, request: Request) -> list[ActionOutput]:
        raise NotImplementedError("Strategy 3: Semantic Classification not yet implemented")
