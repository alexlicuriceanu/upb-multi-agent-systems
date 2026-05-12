"""RequestSolver agent for handling smart home requests."""

import threading
from models import Request, ActionOutput
from environment_manager import EnvironmentManagerAgent
from agent_protocol import AgentMailbox, MessageType, get_message_broker


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
    """Strategy 1: Full Context – not yet implemented."""

    def solve(self, request: Request) -> list[ActionOutput]:
        raise NotImplementedError("Strategy 1: Full Context not yet implemented")


class SequentialSolver(RequestSolverAgent):
    """Strategy 2: Sequential Exploration – not yet implemented."""

    def solve(self, request: Request) -> list[ActionOutput]:
        raise NotImplementedError("Strategy 2: Sequential Exploration not yet implemented")


class SemanticSolver(RequestSolverAgent):
    """Strategy 3: Semantic Classification – not yet implemented."""

    def solve(self, request: Request) -> list[ActionOutput]:
        raise NotImplementedError("Strategy 3: Semantic Classification not yet implemented")
