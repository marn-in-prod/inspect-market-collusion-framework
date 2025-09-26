# tools/market_info.py
from inspect_ai.tool import tool
from typing import Dict, Any
from config.types import GameState

@tool
def market_info_tool(game_state: GameState, agent_id: str):
    """A tool to provide an agent with their current status and market state."""
    async def execute() -> Dict[str, Any]:
        """
        Get current market information, your cash, and your contract holdings.

        Returns:
            A dictionary containing current market prices for unresolved events,
            your personal portfolio (cash and contracts), and a list of events
            that have already been resolved.
        """
        agent = game_state["agents"][agent_id]

        # Get current prices for unresolved events
        current_prices = {
            event["id"]: game_state["prices"][event["id"]].copy()
            for event in game_state["events"] if not event["resolved"]
        }

        # Get current holdings and calculate portfolio value
        current_holdings = {}
        total_contract_value = 0
        for event_id, contracts in agent["contracts"].items():
            if contracts["yes"] > 0 or contracts["no"] > 0:
                current_holdings[event_id] = contracts.copy()
                if not next((e for e in game_state["events"] if e["id"] == event_id), {}) .get("resolved"):
                    value = (contracts["yes"] * current_prices[event_id]["yes"] +
                             contracts["no"] * current_prices[event_id]["no"])
                    total_contract_value += value

        # Get resolved events
        resolved_events = [
            {"event_id": event["id"], "outcome": "YES" if event["outcome"] else "NO"}
            for event in game_state["events"] if event["resolved"]
        ]

        return {
            "agent_id": agent_id,
            "agent_type": agent["type"],
            "current_round": game_state["round"],
            "cash": agent["cash"],
            "estimated_portfolio_value": agent["cash"] + total_contract_value,
            "current_prices_unresolved": current_prices,
            "holdings": current_holdings,
            "resolved_events": resolved_events,
        }

    return execute