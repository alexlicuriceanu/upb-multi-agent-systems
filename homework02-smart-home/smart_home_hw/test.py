import json
from models import Preference, TimeInterval
from environment_manager import EnvironmentManagerAgent

def load_preferences():
    with open('../preferences.json', 'r') as f:
        data = json.load(f)
    
    prefs = []
    for p in data:
        interval = TimeInterval(start=p['dislike_interval']['start'], end=p['dislike_interval']['end'])
        prefs.append(Preference(
            device_type=p['device_type'],
            room=p['room'],
            dislike_interval=interval,
            reason=p['reason']
        ))
    return prefs

def main():
    print("EnvironmentManager Test")
    prefs = load_preferences()
    
    env_agent = EnvironmentManagerAgent(
        simulator_base_url="http://localhost:8080", 
        preferences=prefs, 
        verbose=True
    )
    
    # Test get_rooms
    print("\n[TEST] Testing get_rooms for 'home12'...")
    rooms = env_agent.get_rooms("home12")
    print(f"Result: {rooms}")

    # Test get_artifacts_in_room
    print("\n[TEST] Testing get_artifacts_in_room for 'home12/guest_bedroom'...")
    artifacts = env_agent.get_artifacts_in_room("home12", "guest_bedroom")
    print(f"Result: {artifacts}")

    # Test get_artifact_affordances (Assuming there is a dehumidifier based on request.json)
    if artifacts:
        target_artifact = artifacts[0]
        print(f"\n[TEST] Testing get_artifact_affordances for {target_artifact}...")
        info = env_agent.get_artifact_affordances(target_artifact)
        print(f"Device Type: {info.device_type}")
        print(f"Actions: {[a.name for a in info.actions]}")
        print(f"Properties: {[p.name for p in info.properties]}")

        # Test get_artifact_state
        print(f"\n[TEST] Testing get_artifact_state for {target_artifact}...")
        state = env_agent.get_artifact_state(target_artifact)
        print(f"State: {state.properties}")

    # Test get_active_preferences
    print("\n[TEST] Testing get_active_preferences at '11:30'...")
    active = env_agent.get_active_preferences("11:30")
    print(f"Result: {[p.reason for p in active]}")

if __name__ == "__main__":
    main()