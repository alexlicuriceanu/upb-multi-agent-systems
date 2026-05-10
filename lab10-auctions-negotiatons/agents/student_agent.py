from typing import List, Dict, Any

from scipy.stats._multivariate import special_ortho_group_frozen

from agents import HouseOwnerAgent, CompanyAgent
from communication import NegotiationMessage


class MyACMEAgent(HouseOwnerAgent):

    def __init__(self, role: str, budget_list: List[Dict[str, Any]]):
        super(MyACMEAgent, self).__init__(role, budget_list)
        self.auction_prices = {}

    def propose_item_budget(self, auction_item: str, auction_round: int) -> float:
        max_budget = self.budget_dict[auction_item]
        
        # Start low and increment up to the max budget across 3 rounds
        if auction_round == 0:
            offer = max_budget * 0.6
        elif auction_round == 1:
            offer = max_budget * 0.8
        else:
            offer = max_budget * 1.0
            
        self.auction_prices[auction_item] = offer
        return offer

    def notify_auction_round_result(self, auction_item: str, auction_round: int, responding_agents: List[str]):
        pass

    def provide_negotiation_offer(self, negotiation_item: str, partner_agent: str, negotiation_round: int) -> float:
        auction_price = self.auction_prices.get(negotiation_item, self.budget_dict[negotiation_item])
        
        if negotiation_round == 0:
            offer = auction_price * 0.8
        elif negotiation_round == 1:
            offer = auction_price * 0.9
        else:
            offer = auction_price * 1.0
            
        print(f"  -> ACME offers {offer} to {partner_agent} for {negotiation_item} (Round {negotiation_round})")
        return offer

    def notify_partner_response(self, response_msg: NegotiationMessage) -> None:
        pass

    def notify_negotiation_winner(self, negotiation_item: str, winning_agent: str, winning_offer: float) -> None:
        pass


class MyCompanyAgent(CompanyAgent):

    def __init__(self, role: str, specialties: List[Dict[str, Any]]):
        super(MyCompanyAgent, self).__init__(role, specialties)
        self.auction_prices = {}
        self.last_offers = {}

    def decide_bid(self, auction_item: str, auction_round: int, item_budget: float) -> bool:
        cost = self.specialties.get(auction_item)
        
        # If the company doesn't have this specialty, don't bid
        if cost is None:
            return False
            
        # Only bid if the proposed budget covers our cost
        if item_budget >= cost:
            self.auction_prices[auction_item] = item_budget
            return True
            
        return False

    def notify_won_auction(self, auction_item: str, auction_round: int, num_selected: int):
        pass

    def respond_to_offer(self, initiator_msg: NegotiationMessage) -> float:
        auction_item = initiator_msg.negotiation_item
        cost = self.specialties[auction_item]
        auction_price = self.auction_prices.get(auction_item, cost * 1.5)
        round_num = initiator_msg.round

        if round_num == 0:
            offer = auction_price
        elif round_num == 1:
            offer = cost + (auction_price - cost) * 0.5
            if offer >= auction_price:
                offer = auction_price - 1
        else:
            offer = cost
            prev_offer = self.last_offers.get(auction_item, auction_price)
            if offer >= prev_offer:
                offer = prev_offer - 1

        self.last_offers[auction_item] = offer
        
        print(f"  <- {self.name} counters with {offer} for {auction_item} (Round {round_num})")
        return offer

    def notify_contract_assigned(self, construction_item: str, price: float) -> None:
        pass

    def notify_negotiation_lost(self, construction_item: str) -> None:
        pass
