from environment import *
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import time


class AgentDesire(ABC):
    """
    A structured sub-goal (DESIRE) used by the BDI loop.
    Students should select among available desires and build an intention for one desire at a time.

    Students are encouraged to create subclasses with their own logic for:
      - when the desire is achieved
      - when the desire becomes impossible
      - which blocks are relevant for this desire
    """

    def __init__(self, desire_id: str, description: str):
        self.desire_id = desire_id
        self.description = description


    @abstractmethod
    def is_achieved(self, current_world: BlocksWorld, holding_block: Optional[Block] = None) -> bool:
        raise NotImplementedError()


    @abstractmethod
    def is_impossible(self, current_world: BlocksWorld, holding_block: Optional[Block] = None) -> bool:
        raise NotImplementedError()


    @abstractmethod
    def get_desired_blocks(self) -> List[Block]:
        raise NotImplementedError()


    def __str__(self) -> str:
        return f"{self.desire_id}: {self.description}"


@dataclass
class PlaceBlockDesire(AgentDesire):
    block: Block
    support: Optional[Block]

    def __init__(self, block: Block, support: Optional[Block]):
        if support is None:
            description = f"Place {block} on table"
        else:
            description = f"Place {block} on {support}"

        super().__init__(desire_id=f"place-{block}", description=description)
        self.block = block
        self.support = support


    def is_achieved(self, current_world: BlocksWorld, holding_block: Optional[Block] = None) -> bool:
        try:
            stack = current_world.get_stack(self.block)
            if self.support is None:
                return stack.is_on_table(self.block)
            return stack.is_on(self.block, self.support)
        except Exception:
            return False


    def is_impossible(self, current_world: BlocksWorld, holding_block: Optional[Block] = None) -> bool:
        try:
            stack = current_world.get_stack(self.block)
        except Exception:
            return holding_block != self.block

        if self.support is not None:
            try:
                current_world.get_stack(self.support)
            except Exception:
                return holding_block != self.support

        if stack.is_locked(self.block) and not self.is_achieved(current_world, holding_block):
            return True

        return False


    def get_desired_blocks(self) -> List[Block]:
        return [self.block]

@dataclass
class BuildStackDesire(AgentDesire):
    stack_blocks: List[Block]

    def __init__(self, stack_blocks: List[Block]):
        super().__init__(
            desire_id="stack-" + "-".join([str(b) for b in stack_blocks]),
            description="Build stack: " + "-".join([str(b) for b in stack_blocks]),
        )
        self.stack_blocks = list(stack_blocks)

    def is_achieved(self, current_world: BlocksWorld, holding_block: Optional[Block] = None) -> bool:
        if len(self.stack_blocks) == 0:
            return False

        try:
            bottom = self.stack_blocks[0]
            stack = current_world.get_stack(bottom)
            if not stack.is_on_table(bottom):
                return False

            for idx in range(1, len(self.stack_blocks)):
                if not stack.is_on(self.stack_blocks[idx], self.stack_blocks[idx - 1]):
                    return False

            return True
        except Exception:
            return False

    def is_impossible(self, current_world: BlocksWorld, holding_block: Optional[Block] = None) -> bool:
        if len(self.stack_blocks) == 0:
            return True

        if self.is_achieved(current_world, holding_block):
            return False

        for idx, b in enumerate(self.stack_blocks):
            try:
                block_stack = current_world.get_stack(b)
                if block_stack.is_locked(b):
                    expected_support = None if idx == 0 else self.stack_blocks[idx - 1]
                    if expected_support is None:
                        if not block_stack.is_on_table(b):
                            return True
                    else:
                        try:
                            if not block_stack.is_on(b, expected_support):
                                return True
                        except ValueError:
                            # the support block isn't in this stack, so it's definitely in the wrong place
                            return True
            except Exception:
                # fix: if the block is not in a stack, it's either held by the arm or stashed by the environment.
                # in neither case does the desire become permanently impossible, we just pass
                pass

        return False

    def get_desired_blocks(self) -> List[Block]:
        return list(self.stack_blocks)

@dataclass
class BuildRowDesire(AgentDesire):
    row_blocks: List[Block]
    row_level: int

    def __init__(self, row_blocks: List[Block], row_level: int):
        super().__init__(
            desire_id=f"row-{row_level}-" + "-".join([str(b) for b in row_blocks]),
            description=f"Build row at level {row_level}: " + "-".join([str(b) for b in row_blocks]),
        )
        self.row_blocks = list(row_blocks)
        self.row_level = row_level


    def is_achieved(self, current_world: BlocksWorld, holding_block: Optional[Block] = None) -> bool:
        if len(self.row_blocks) == 0:
            return False

        used_stack_bases = set()
        for block in self.row_blocks:
            try:
                stack = current_world.get_stack(block)
            except Exception:
                return False

            blocks = stack.get_blocks()
            if block not in blocks:
                return False

            if blocks.index(block) != self.row_level:
                return False

            stack_base = stack.get_bottom_block()
            if stack_base in used_stack_bases:
                return False

            used_stack_bases.add(stack_base)

        return True


    def is_impossible(self, current_world: BlocksWorld, holding_block: Optional[Block] = None) -> bool:
        if len(self.row_blocks) == 0:
            return True

        stack_to_row_blocks = {}
        for b in self.row_blocks:
            try:
                block_stack = current_world.get_stack(b)
            except Exception:
                if holding_block != b:
                    return True
                continue

            if block_stack.is_locked(b):
                block_level = block_stack.get_blocks().index(b)
                if block_level != self.row_level:
                    return True

            stack_base = block_stack.get_bottom_block()
            if stack_base not in stack_to_row_blocks:
                stack_to_row_blocks[stack_base] = []
            stack_to_row_blocks[stack_base].append(b)

        if self.is_achieved(current_world, holding_block):
            return False

        for stack_base, row_blocks_in_stack in stack_to_row_blocks.items():
            if len(row_blocks_in_stack) <= 1:
                continue

            locked_row_targets_in_same_stack = []
            for b in row_blocks_in_stack:
                try:
                    if current_world.get_stack(b).is_locked(b):
                        locked_row_targets_in_same_stack.append(b)
                except Exception:
                    return True

            # Row semantics require target row blocks to end up on distinct stacks.
            # If at least two row-target blocks are already locked in the same stack,
            # they cannot be separated anymore, so the row desire is impossible.
            # Note: this does NOT count locked support blocks under the row level.
            if len(locked_row_targets_in_same_stack) >= 2:
                return True

        return False


    def get_desired_blocks(self) -> List[Block]:
        return list(self.row_blocks)

class MyAgent(BlocksWorldAgent):

    MODE_IDLE = "IDLE"
    MODE_COMMITTED = "COMMITTED"

    def __init__(self, name: str, target_state: BlocksWorld):
        super(MyAgent, self).__init__(name=name)

        self.target_state = target_state

        """
        The agent's belief about the world state. Initially, the agent has no belief about the world state.
        """
        self.belief: BlocksWorld = None

        """
        The agent's current desire. It is expressed as a list of blocks for which the agent wants to make a plan to bring to their corresponding
        configuration in the target state. 
        The list can contain a single block or a sequence of blocks that represent: (i) a stack of blocks, (ii) a row of blocks (e.g. going level by level).
        """
        self.current_desire: Optional[AgentDesire] = None

        """
        The set of possible desires (sub-goals) extracted from the target world.
        One desire corresponds to putting one block into its target relation (on table or on another block).
        """
        self.desire_pool: List[AgentDesire] = []

        """
        The current intention is the agent plan (sequence of actions) that the agent is executing to achieve the current desire.
        """
        self.current_intention: List[BlocksWorldAction] = []

        self.mode: str = MyAgent.MODE_IDLE
        self.last_action: Optional[BlocksWorldAction] = None
        self.last_failure_reason: Optional[str] = None

        self._initialize_desire_pool_from_target()


    def response(self, perception: BlocksWorldPerception) -> BlocksWorldAction:
        ## if the perceived state contains the target state, the agent has achieved its goal
        if perception.current_world.contains_world(self.target_state):
            return AgentCompleted()
        
        ## revise the agents beliefs based on the perceived state
        self.revise_beliefs(
            perception.current_world,
            perception.previous_action_succeeded,
            perception.previous_action_message,
        )

        if self.mode == MyAgent.MODE_COMMITTED and self.current_desire:
            if self._is_desire_achieved(self.current_desire, perception.current_world, perception.holding_block):
                self._drop_current_desire("desire achieved")
            elif self._is_desire_impossible(self.current_desire, perception.current_world, perception.holding_block):
                self._drop_current_desire("desire became impossible in current world")

        if self.mode == MyAgent.MODE_IDLE:
            selected_desire = self._select_next_desire(perception.current_world, perception.holding_block)
            if selected_desire is None:
                self.last_failure_reason = "NO_DESIRE_SELECTED"
                self.last_action = NoAction()
                return self.last_action

            self.current_desire = selected_desire
            self.current_intention = []
            self.mode = MyAgent.MODE_COMMITTED

        if self.mode == MyAgent.MODE_COMMITTED and self.current_desire and len(self.current_intention) == 0:
            self.current_intention = self._plan_for_current_desire(perception.current_world, perception.holding_block)
            if len(self.current_intention) == 0:
                self._drop_current_desire("no plan available for committed desire")
                self.last_action = NoAction()
                return self.last_action

        if self.mode == MyAgent.MODE_COMMITTED and self.current_desire and len(self.current_intention) > 0:
            next_action = self.current_intention[0]

            if self._can_apply_action(next_action, perception.current_world, perception.holding_block):
                self.last_action = self.current_intention.pop(0)
                return self.last_action

            self.current_intention = self._plan_for_current_desire(perception.current_world, perception.holding_block)
            if len(self.current_intention) == 0:
                self._drop_current_desire("intention invalidated and replan failed")
                self.last_action = NoAction()
                return self.last_action

            next_action = self.current_intention[0]
            if self._can_apply_action(next_action, perception.current_world, perception.holding_block):
                self.last_action = self.current_intention.pop(0)
                return self.last_action

            self._drop_current_desire("replanned first action still not applicable")

        self.last_action = NoAction()
        return self.last_action


    def _can_apply_action(self, act: BlocksWorldAction, world: BlocksWorld, holding_block: Optional[Block]) -> bool:
        """
        Check if the action can be applied to the current world state.
        """
        ## create a clone of the world
        sim_world = world.clone()

        ## apply the action to the clone, surrpressing any exceptions
        try:
            ## locking can be performed at any time, so check if the action is a lock actio
            if act.get_type() == "lock":
                ## try to lock the block
                sim_world.lock(act.get_argument())
            else:
                if holding_block is None:
                    if act.get_type() == "putdown" or act.get_type() == "stack":
                        ## If we are not holding anything, we cannot putdown or stack a block
                        return False
                    
                    if act.get_type() == "pickup":
                        ## try to pickup the block
                        sim_world.pick_up(act.get_argument())
                    elif act.get_type() == "unstack":
                        ## try to unstack the block
                        sim_world.unstack(act.get_first_arg(), act.get_second_arg())
                else:
                    ## we are holding a block, so we can only putdown or stack
                    if act.get_type() == "pickup" or act.get_type() == "unstack":
                        ## If we are holding a block, we cannot pickup or unstack
                        return False

                    if act.get_type() == "putdown":
                        ## If we want to putdown the block we have to check if it's the same block we are holding
                        if act.get_argument() != holding_block:
                            return False
                        sim_world.put_down(act.get_argument(), sim_world.get_stacks()[-1])

                    if act.get_type() == "stack":
                        ## If we want to stack the block we have to check if it's the same block we are holding
                        if act.get_first_arg() != holding_block:
                            return False
                        ## try to stack the block
                        sim_world.stack(act.get_first_arg(), act.get_second_arg())
        except Exception as e:
            return False
        
        return True


    def _initialize_desire_pool_from_target(self) -> None:
        """
        TODO (student): build the initial list of desires from the target world.
        Examples of desire subclasses that can be added:
          - PlaceBlockDesire
          - BuildStackDesire
          - BuildRowDesire

        The method should populate `self.desire_pool` with all desires that the student strategy may consider.
        """
        self.desire_pool = []

        for stack in self.target_state.get_stacks():
            blocks_in_stack = list(stack.get_blocks())

            if blocks_in_stack:
                self.desire_pool.append(BuildStackDesire(blocks_in_stack))


    def _drop_current_desire(self, reason: str) -> None:
        if self.current_desire is not None:
            self.last_failure_reason = f"{reason} | desire={self.current_desire.desire_id} ({self.current_desire.description})"
        else:
            self.last_failure_reason = reason
        self.current_desire = None
        self.current_intention = []
        self.mode = MyAgent.MODE_IDLE


    def _select_next_desire(self, current_world: BlocksWorld, holding_block: Optional[Block] = None) -> Optional[AgentDesire]:
        """
        TODO (student): select one desire from the available/required desires.
        This method must return exactly one desire (or None if none can be pursued right now).
        """
        for desire in self.desire_pool:
            if not isinstance(desire, BuildStackDesire):
                continue
            
            if desire.is_achieved(current_world, holding_block):
                continue

            if desire.is_impossible(current_world, holding_block):
                continue

            # we could also pick the one that is closest to completion
            return desire

        return None


    def _plan_for_current_desire(self, current_world: BlocksWorld, holding_block: Optional[Block]) -> List[BlocksWorldAction]:
        """
        TODO (student): build a plan ONLY for the currently committed desire (not for full goal).
        Return a list of actions that should be executed until the desire is achieved or impossible.
        """
        if not isinstance(self.current_desire, BuildStackDesire):
            return []
        
        target_stack = self.current_desire.stack_blocks
        
        for i, block in enumerate(target_stack):
            required_support = target_stack[i-1] if i > 0 else None
            
            try:
                current_stack = current_world.get_stack(block)
            except ValueError:
                current_stack = None

            is_in_position = False
            
            if current_stack:
                if required_support is None:
                    if current_stack.is_on_table(block):
                        is_in_position = True
                else:
                    try:
                        if current_stack.get_below(block) == required_support:
                            is_in_position = True
                    except ValueError:
                        pass
            
            # Case A: Block is in correct position
            if is_in_position:
                if not current_stack.is_locked(block):
                    if current_stack.is_clear(block):
                        return [Lock(block)]
                    else:
                        block_above = current_stack.get_above(block)
                        return [Unstack(block_above, block)]
                continue

            # Case B: Block is NOT in position
            if holding_block != block:
                if holding_block:
                    return [PutDown(holding_block)]
                
                if not current_stack: return [] 
                
                if current_stack.is_clear(block):
                    if current_stack.is_on_table(block):
                        return [PickUp(block)]
                    else:
                        return [Unstack(block, current_stack.get_below(block))]
                else:
                    block_above = current_stack.get_above(block)
                    return [Unstack(block_above, block)]
            
            else: # Holding the block
                if required_support is None:
                    return [PutDown(block)]
                else:
                    try:
                        supp_stack = current_world.get_stack(required_support)
                        # SAFE CHECK: Use get_blocks()[-1] instead of is_clear()
                        # This avoids crashing if the support stack is fully locked.
                        all_blocks = supp_stack.get_blocks()
                        if all_blocks and all_blocks[-1] == required_support:
                            return [Stack(block, required_support)]
                        else:
                            return [PutDown(block)]
                    except ValueError:
                        return [PutDown(block)]
        
        return []


    def _is_desire_achieved(self, desire: AgentDesire, current_world: BlocksWorld, holding_block: Optional[Block] = None) -> bool:
        try:
            return desire.is_achieved(current_world, holding_block)
        except NotImplementedError:
            return False


    def _is_desire_impossible(self, desire: AgentDesire, current_world: BlocksWorld, holding_block: Optional[Block] = None) -> bool:
        try:
            return desire.is_impossible(current_world, holding_block)
        except NotImplementedError:
            return True


    def revise_beliefs(self, perceived_world_state: BlocksWorld, previous_action_succeeded: bool,
                      previous_action_message: Optional[str] = None):
        """
        TODO: revise internal agent structured depending on whether what the agent *expects* to be true 
        corresponds to what the agent perceives from the environment.
        :param perceived_world_state: the world state perceived by the agent
        :param previous_action_succeeded: whether the previous action succeeded or not
        """
        self.belief = perceived_world_state.clone()

        if previous_action_succeeded:
            if previous_action_message:
                self.last_failure_reason = f"last outcome: {previous_action_message}"
            return

        if self.mode == MyAgent.MODE_COMMITTED and self.current_desire:
            self.last_failure_reason = (
                f"last action failed while committed to {self.current_desire.desire_id}"
                + (f" | {previous_action_message}" if previous_action_message else "")
            )
        elif previous_action_message:
            self.last_failure_reason = f"last action failed | {previous_action_message}"


    def plan(self) -> Tuple[List[Block], List[BlocksWorldAction]]:
        """
        Deprecated compatibility shim for old lab skeleton.
        Planning must now be done through `_select_next_desire` and `_plan_for_current_desire`.
        """
        selected = self._select_next_desire(self.belief if self.belief is not None else self.target_state, None)
        if selected is None:
            return [], []
        self.current_desire = selected
        return selected.get_desired_blocks(), self._plan_for_current_desire(self.belief if self.belief is not None else self.target_state, None)


    def status_string(self):
        desire_info = "none" if self.current_desire is None else self.current_desire.description
        next_action = "none" if len(self.current_intention) == 0 else str(self.current_intention[0])
        return (
            f"{self} : MODE={self.mode} | DESIRE={desire_info} | "
            f"INTENTION_STEPS={len(self.current_intention)} | NEXT={next_action} | "
            f"LAST={self.last_action} | REASON={self.last_failure_reason}"
        )



class Tester(object):
    STEP_DELAY = 0.0
    TEST_SUITE = "tests/0e-large/"
    VERBOSE = True

    EXT = ".txt"
    SI  = "si"
    SF  = "sf"

    DYNAMICS_PROB = 1.0

    AGENT_NAME = "*A"

    def __init__(self):
        self._environment = None
        self._agents = []

        self._initialize_environment(Tester.TEST_SUITE)
        self._initialize_agents(Tester.TEST_SUITE)



    def _initialize_environment(self, test_suite: str) -> None:
        filename = test_suite + Tester.SI + Tester.EXT

        with open(filename) as input_stream:
            self._environment = DynamicEnvironment(
                BlocksWorld(input_stream=input_stream),
                verbose=Tester.VERBOSE,
                dynamics_prob=Tester.DYNAMICS_PROB,
            )


    def _initialize_agents(self, test_suite: str) -> None:
        filename = test_suite + Tester.SF + Tester.EXT

        agent_states = {}

        with open(filename) as input_stream:
            desires = BlocksWorld(input_stream=input_stream)
            agent = MyAgent(Tester.AGENT_NAME, desires)

            agent_states[agent] = desires
            self._agents.append(agent)

            self._environment.add_agent(agent, desires, None)

            if Tester.VERBOSE:
                print("Agent %s desires:" % str(agent))
                print(str(desires))


    def make_steps(self):
        if Tester.VERBOSE:
            print("\n\n================================================= INITIAL STATE:")
            print(str(self._environment))
            print("\n\n=================================================")
        else:
            print("Simulation started (verbose=False)")

        completed = False
        nr_steps = 0

        while not completed:
            completed = self._environment.step()

            time.sleep(Tester.STEP_DELAY)
            if Tester.VERBOSE:
                print(str(self._environment))

                for ag in self._agents:
                    print(ag.status_string())

            nr_steps += 1

            if Tester.VERBOSE:
                print("\n\n================================================= STEP %i completed." % nr_steps)

        if Tester.VERBOSE:
            print("\n\n================================================= ALL STEPS COMPLETED")
        else:
            print("Simulation completed in %i steps" % nr_steps)





if __name__ == "__main__":
    tester = Tester()
    tester.make_steps()