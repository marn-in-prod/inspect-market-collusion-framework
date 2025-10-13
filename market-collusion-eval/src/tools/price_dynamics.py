# tools/price_dynamics.py - Dynamic price updates based on trading volume
from config.types import GameState
from typing import Dict, Any
import math

class PricingEngine:
    """Manages dynamic price updates based on market activity."""
    
    def __init__(self, sensitivity: float = 0.05, max_price_change: float = 0.15):
        """
        Initialize pricing engine.
        
        Args:
            sensitivity: How much prices respond to trades (0.01-0.10 typical)
            max_price_change: Maximum price change per round as fraction (e.g., 0.15 = 15%)
        """
        self.sensitivity = sensitivity
        self.max_price_change = max_price_change
        self.trade_history = {}  # Track cumulative trades per event
    
    def initialize_trade_tracking(self, game_state: GameState) -> None:
        """Initialize trade tracking for all events."""
        for event in game_state["events"]:
            event_id = event["id"]
            self.trade_history[event_id] = {
                "yes_volume": 0,
                "no_volume": 0,
                "total_volume": 0,
                "last_update_round": 0
            }
    
    def update_prices_from_trades(self, game_state: GameState) -> Dict[str, Any]:
        """
        Update market prices based on recent trading activity.
        
        Returns:
            Dictionary with price changes and statistics
        """
        if not hasattr(self, 'trade_history') or not self.trade_history:
            self.initialize_trade_tracking(game_state)
        
        current_round = game_state["round"]
        config = game_state["config"]
        price_changes = {}
        
        # Get trades from this round
        round_trades = [t for t in game_state["trade_log"] if t["round"] == current_round]
        
        # Group trades by event
        event_trades = {}
        for trade in round_trades:
            event_id = trade["event_id"]
            if event_id not in event_trades:
                event_trades[event_id] = {"yes": 0, "no": 0}
            
            # Count net buying pressure (buys are positive, sells are negative)
            if trade["action"] == "buy":
                event_trades[event_id][trade["side"]] += trade["quantity"]
            else:  # sell
                event_trades[event_id][trade["side"]] -= trade["quantity"]
        
        # Update prices for each event with trades
        for event_id, net_trades in event_trades.items():
            # Skip resolved events
            event = next((e for e in game_state["events"] if e["id"] == event_id), None)
            if not event or event["resolved"]:
                continue
            
            old_prices = game_state["prices"][event_id].copy()
            
            # Calculate new implied probability
            net_yes = net_trades["yes"]
            net_no = net_trades["no"]
            total_net_volume = abs(net_yes) + abs(net_no)
            
            if total_net_volume == 0:
                continue  # No net change
            
            # Calculate current implied probability from prices
            current_yes_price = old_prices["yes"]
            current_prob = current_yes_price / config.market.contract_payout
            
            # Calculate price pressure based on net trades
            # Positive net_yes increases probability, negative decreases it
            price_pressure = (net_yes - net_no) / (total_net_volume + 10)  # +10 to prevent extreme moves
            price_adjustment = price_pressure * self.sensitivity
            
            # Cap the adjustment
            price_adjustment = max(-self.max_price_change, min(self.max_price_change, price_adjustment))
            
            # Calculate new probability
            new_prob = current_prob + price_adjustment
            new_prob = max(0.05, min(0.95, new_prob))  # Keep within bounds
            
            # Calculate new prices
            new_yes_price = new_prob * config.market.contract_payout
            new_no_price = (1 - new_prob) * config.market.contract_payout
            
            # Update game state
            game_state["prices"][event_id]["yes"] = round(new_yes_price, 2)
            game_state["prices"][event_id]["no"] = round(new_no_price, 2)
            
            # Update trade history
            self.trade_history[event_id]["yes_volume"] += abs(net_yes)
            self.trade_history[event_id]["no_volume"] += abs(net_no)
            self.trade_history[event_id]["total_volume"] += total_net_volume
            self.trade_history[event_id]["last_update_round"] = current_round
            
            # Record the change
            price_changes[event_id] = {
                "old_yes": old_prices["yes"],
                "new_yes": game_state["prices"][event_id]["yes"],
                "old_no": old_prices["no"],
                "new_no": game_state["prices"][event_id]["no"],
                "prob_change": price_adjustment,
                "net_yes_trades": net_yes,
                "net_no_trades": net_no
            }
        
        return price_changes
    
    def get_market_depth(self, game_state: GameState, event_id: str) -> Dict[str, float]:
        """Get trading volume statistics for an event."""
        if event_id not in self.trade_history:
            return {"yes_volume": 0, "no_volume": 0, "total_volume": 0}
        
        return self.trade_history[event_id].copy()

# Global pricing engine
pricing_engine = PricingEngine(sensitivity=0.05, max_price_change=0.15)

def update_market_prices(game_state: GameState) -> str:
    """
    Update all market prices based on trading activity.
    Called at the end of each round.
    
    Returns:
        Summary of price changes
    """
    price_changes = pricing_engine.update_prices_from_trades(game_state)
    
    if not price_changes:
        return "No significant price movements this round."
    
    summary = "Market Price Updates:\n"
    for event_id, changes in price_changes.items():
        yes_change = changes["new_yes"] - changes["old_yes"]
        yes_pct = (yes_change / changes["old_yes"]) * 100 if changes["old_yes"] > 0 else 0
        
        summary += f"  {event_id}: YES ${changes['old_yes']:.2f} → ${changes['new_yes']:.2f} "
        summary += f"({yes_pct:+.1f}%) | NO ${changes['old_no']:.2f} → ${changes['new_no']:.2f}\n"
        summary += f"    Net trades: {changes['net_yes_trades']:+d} YES, {changes['net_no_trades']:+d} NO\n"
    
    return summary.strip()

def initialize_pricing_engine(game_state: GameState) -> None:
    """Initialize the pricing engine for a new game."""
    pricing_engine.initialize_trade_tracking(game_state)

def get_pricing_analytics(game_state: GameState) -> Dict[str, Any]:
    """Get analytics on price movements."""
    analytics = {
        "total_events": len(game_state["events"]),
        "events_traded": 0,
        "total_volume": 0,
        "most_traded_event": None,
        "max_volume": 0
    }
    
    for event_id, history in pricing_engine.trade_history.items():
        if history["total_volume"] > 0:
            analytics["events_traded"] += 1
            analytics["total_volume"] += history["total_volume"]
            
            if history["total_volume"] > analytics["max_volume"]:
                analytics["max_volume"] = history["total_volume"]
                analytics["most_traded_event"] = event_id
    
    return analytics