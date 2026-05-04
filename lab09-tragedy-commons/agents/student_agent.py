from typing import Callable, List, Dict

from commons import CommonsAgent, CommonsPerception
from communication import AgentAction

class StudentAgent(CommonsAgent):
    def __init__(self, agent_id):
        super(StudentAgent, self).__init__(agent_id)
        self.target_total_share = 0.5
        self.current_proposed_share = None

    def specify_share(self, perception: CommonsPerception) -> float:
        ## TODO: return the share that this agent wants to consume at a start of a environment turn
        num_agents = perception.num_agents
        fair_share = self.target_total_share / num_agents
        
        if self.current_proposed_share is None:
            deviation = (self.id - (num_agents + 1) / 2) * 0.03
            self.current_proposed_share = fair_share + deviation
            
        return max(0.01, self.current_proposed_share)

    def negotiation_response(self, negotiation_round: int, perception: CommonsPerception,
                             utility_func: Callable[[float, float, List[float]], float]) -> AgentAction:
        # TODO: return an AgentAction, whereby the agent can specify what his revised consumption share is, as
        # well as what he thinks other agents should consume, in the form of a consumption_adjustment dict
        #
        # Attention: you must pay attention to the fact the the consumption_adjustment dict may indicate altering
        # the agent_shares in such a way that their sum is greater than 1 or smaller than 0. 
        # You must avoid this, as it will lead to the consumption round being aborted and all agents receiving a 0 utility.

        num_agents = perception.num_agents
        fair_share = self.target_total_share / num_agents
        
        my_current_share = perception.resource_shares[self.id]
        
        adjustments: Dict[int, float] = {}
        needs_action = False
        
        # Respond to social inequalities
        for agent_id, share in perception.resource_shares.items():
            # Lowered the deadzone threshold significantly so they fully converge
            if abs(share - fair_share) > 0.00001:
                needs_action = True

                # Use a partial adjustment (0.4) to create a smooth visual curve 
                # on the plot over multiple rounds, rather than snapping instantly.
                adjustments[agent_id] = (fair_share - share) * 0.4
                
        # Avoid total depletion
        total_shares = sum(perception.resource_shares.values())
        if total_shares >= 0.95:
            for agent_id, share in perception.resource_shares.items():
                adjustments[agent_id] = -share * 0.5
            needs_action = True

        if not needs_action:
            return AgentAction(self.id, resource_share=my_current_share, no_action=True)
            
        # Pass our current share unchanged, relying entirely on the 
        # consumption_adjustment to fix the inequalities. This prevents double-correcting.
        return AgentAction(self.id, resource_share=my_current_share, 
                           consumption_adjustment=adjustments, no_action=False)

    def inform_round_finished(self, negotiation_round: int, perception: CommonsPerception):
        ## information sent to the agent once the current round (including all adjustment rounds) is finished
        self.current_proposed_share = perception.resource_shares[self.id]