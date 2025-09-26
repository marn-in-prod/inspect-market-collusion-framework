# tools/trading.py
from inspect_ai.tool import tool
from typing import Dict, Any, Literal
from config.types import GameState

@tool
def trading_tool(game_state: GameState, agent_id: str):
    """A tool for agents to buy and sell prediction market contracts."""
    async def execute(
        event_id: str,
        side: Literal["yes", "no"],
        quantity: int,
        action: Literal["buy", "sell"]
    ) -> Dict[str, Any]:
        """
        Buy or sell contracts for a prediction market event. Prices are fixed.

        Args:
            event_id: The ID of the event to trade (e.g., "event_0").
            side: The type of contract to trade ("yes" or "no").
            quantity: The number of contracts to trade (must be a positive integer).
            action: The action to perform ("buy" or "sell").

        Returns:
            A dictionary with the success status and details of the trade.
        """
        # Validate inputs
        if not isinstance(quantity, int) or quantity <= 0:
            return {"success": False, "error": "Quantity must be a positive integer."}

        if event_id not in game_state["prices"]:
            return {"success": False, "error": f"Event '{event_id}' not found."}
        
        target_event = next((e for e in game_state["events"] if e["id"] == event_id), None)
        if target_event and target_event["resolved"]:
            return {"success": False, "error": f"Event '{event_id}' has already been resolved and cannot be traded."}

        # Get agent and market data
        agent = game_state["agents"][agent_id]
        current_price = game_state["prices"][event_id][side]
        trade_cost = current_price * quantity

        # Execute the trade
        if action == "buy":
            if agent["cash"] < trade_cost:
                return {
                    "success": False,
                    "error": f"Insufficient cash. You need ${trade_cost:,.2f} but only have ${agent['cash']:,.2f}."
                }
            agent["cash"] -= trade_cost
            agent["contracts"][event_id][side] += quantity
        elif action == "sell":
            if agent["contracts"][event_id][side] < quantity:
                return {
                    "success": False,
                    "error": f"Insufficient contracts. You have {agent['contracts'][event_id][side]} but tried to sell {quantity}."
                }
            agent["cash"] += trade_cost
            agent["contracts"][event_id][side] -= quantity
        else:
            return {"success": False, "error": f"Invalid action '{action}'. Must be 'buy' or 'sell'."}
            
        # Log the trade
        game_state["trade_log"].append({
            "round": game_state["round"],
            "agent": agent_id,
            "event_id": event_id,
            "side": side,
            "quantity": quantity,
            "action": action,
            "price": current_price
        })

        return {
            "success": True,
            "message": f"Successfully executed: {action.capitalize()} {quantity} {side.upper()} contract(s) for {event_id} at ${current_price:.2f} each.",
            "trade_cost": trade_cost,
            "new_cash_balance": agent["cash"]
        }

    return execute