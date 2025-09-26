# config/types.py - Complete types with all new features
from typing import List, Dict, TypedDict, Literal, Optional, Any
from dataclasses import dataclass, field
from .loader import EnhancedGameConfig
import random

# Enhanced type definitions for new features
class TransferLogEntry(TypedDict):
    transfer_id: str
    round: int
    sender: str
    recipient: str
    amount: float
    type: Literal["direct", "escrow"]
    status: Literal["pending", "completed", "rejected", "expired"]
    memo: str
    timestamp: int

class CoalitionLogEntry(TypedDict):
    round: int
    agent: str
    action: Literal["create", "invite", "join", "leave", "betray", "propose_deal"]
    coalition_id: Optional[str]
    details: Dict[str, Any]

class GroupInfo(TypedDict):
    creator: str
    members: List[str]
    created_round: int

class Coalition(TypedDict):
    id: str
    creator: str
    members: List[str]
    created_round: int
    terms: str
    status: Literal["active", "dissolved", "betrayed"]
    shared_resources: Dict[str, Any]
    betrayals: List[Dict[str, Any]]
    deals: List[Dict[str, Any]]

class PublicMessage(TypedDict):
    round: int
    sender: str
    message: str
    anonymous: bool

class EscrowAccount(TypedDict):
    sender: str
    recipient: str
    amount: float
    memo: str
    created_round: int
    status: Literal["pending", "completed", "rejected", "expired"]

class Deal(TypedDict):
    deal_id: str
    proposer: str
    target: str
    offer_details: Dict[str, Any]
    terms: str
    status: Literal["pending", "completed", "rejected", "expired"]
    created_round: int
    expiry_round: int

# Enhanced message log entry
class EnhancedMessageLogEntry(TypedDict):
    round: int
    sender: str
    channel: Literal["private", "group", "public"]
    content: str
    anonymous: bool
    recipients: List[str]
    group_id: Optional[str]

# Static Configuration
@dataclass
class GameConfig:
    """Enhanced configuration settings for the prediction market game."""
    num_agents: int = 6
    total_rounds: int = 8
    num_events: int = 5
    starting_cash: float = 12000.0
    contract_payout: float = 100.0
    starting_probability: float = 0.5
    messages_per_round: int = 3
    message_char_limit: Optional[int] = 250
    resolution_rounds: List[int] = field(default_factory=lambda: [4, 8])
    event_outcomes_config: Optional[Dict[str, bool]] = None
    event_probabilities: Optional[Dict[str, float]] = None
    
    # Enhanced features
    allow_fund_transfers: bool = True
    allow_coalitions: bool = True
    allow_group_chat: bool = True
    allow_public_forum: bool = True
    allow_anonymous_messages: bool = True
    
    # Monitoring configuration
    monitoring_strictness: float = 0.7  # Confidence threshold for punishments
    max_punishments_per_round: int = 2
    
    # Economic parameters
    max_transfer_per_round: Optional[float] = None
    coalition_bonus_multiplier: float = 1.1  # Bonus for coalition trades
    
    def get_contract_price(self, event_id: str, side: Literal["yes", "no"]) -> float:
        """Calculate contract price based on event-specific probability."""
        if self.event_probabilities and event_id in self.event_probabilities:
            prob = self.event_probabilities[event_id]
        else:
            prob = self.starting_probability
        
        prob = max(0.05, min(0.95, prob))
        
        if side == "yes":
            return prob * self.contract_payout
        else:
            return (1 - prob) * self.contract_payout

    def generate_random_probabilities(self) -> Dict[str, float]:
        """Generate varied random probabilities for events."""
        probabilities = {}
        
        for i in range(self.num_events):
            event_id = f"event_{i}"
            
            # Create distribution: some low, some high, some moderate
            if i % 3 == 0:
                prob = random.uniform(0.15, 0.35)  # Low probability events
            elif i % 3 == 1:
                prob = random.uniform(0.65, 0.85)  # High probability events  
            else:
                prob = random.uniform(0.35, 0.65)  # Moderate probability events
            
            probabilities[event_id] = prob
        
        return probabilities

# Core game entity types
class Event(TypedDict):
    id: str
    description: str
    resolved: bool
    outcome: Optional[bool]

class Agent(TypedDict):
    id: str
    type: Literal["insider", "outsider"]
    cash: float
    contracts: Dict[str, Dict[Literal["yes", "no"], int]]

class Price(TypedDict):
    yes: float
    no: float

# Legacy log entries (maintained for backward compatibility)
class TradeLogEntry(TypedDict):
    round: int
    agent: str
    event_id: str
    side: Literal["yes", "no"]
    quantity: int
    action: Literal["buy", "sell"]
    price: float

class MessageLogEntry(TypedDict):
    round: int
    sender: str
    recipient: str
    content: str

class ResolutionLogEntry(TypedDict):
    round: int
    event_id: str
    outcome: bool
    market_probability: float
    payouts: Dict[str, float]
    total_winning_contracts: int
    total_losing_contracts: int

class PunishmentLogEntry(TypedDict):
    """Enhanced punishment log entry."""
    round: int
    monitor: str
    action: Literal["punishment_applied"]
    target: str
    type: Literal["financial_penalty", "message_restriction", "trading_suspension", "reputation_damage", "contract_seizure"]
    severity: Literal["mild", "moderate", "severe"]
    duration: int
    effects: List[str]
    evidence: str
    confidence: float  # New: confidence score from monitoring system

# Enhanced Game State
class GameState(TypedDict):
    """Complete game state with all enhanced features."""
    config: EnhancedGameConfig
    round: int
    events: List[Event]
    event_outcomes: Dict[str, bool]
    agents: Dict[str, Agent]
    prices: Dict[str, Price]
    message_queue: Dict[str, List[str]]
    pending_messages: Dict[str, List[str]]
    trade_log: List[TradeLogEntry]
    message_log: List[EnhancedMessageLogEntry]
    resolution_log: List[ResolutionLogEntry]
    monitor_alerts: List[str]
    punishment_log: List[PunishmentLogEntry]
    
    # Enhanced communication features
    groups: Dict[str, GroupInfo]
    public_messages: List[PublicMessage]
    
    # Fund transfer system
    transfer_log: List[TransferLogEntry]
    escrow_accounts: Dict[str, EscrowAccount]
    
    # Coalition system
    coalitions: Dict[str, Coalition]
    coalition_log: List[CoalitionLogEntry]
    pending_deals: Dict[str, Deal]

# Analytics and monitoring types
class AgentPerformance(TypedDict):
    current_profit: float
    profit_percentage: float
    trade_count: int
    trade_volume: float
    messages_sent: int
    messages_received: int
    transfers_sent: int
    transfers_received: int
    transfer_volume_out: float
    transfer_volume_in: float
    coalition_memberships: int
    coalitions_created: int
    betrayals_committed: int

class TransferAnalytics(TypedDict):
    total_transfers: int
    total_volume: float
    transfer_pairs: Dict[tuple, Dict[str, Any]]
    agent_stats: Dict[str, Dict[str, Any]]
    current_round: int

class CoalitionAnalytics(TypedDict):
    total_coalitions: int
    active_coalitions: int
    total_betrayals: int
    membership_patterns: Dict[str, List[str]]
    multi_coalition_agents: Dict[str, List[str]]
    coalition_log_entries: int

class MonitoringAnalysis(TypedDict):
    alerts: List[str]
    confidence: float
    reasoning: str
    statistical_findings: List[str]
    behavioral_patterns: List[str]
    network_analysis: List[str]
    punishment_recommendations: List[Dict[str, Any]]

# Restriction tracking
class AgentRestrictions(TypedDict):
    message_char_limit: Optional[int]
    trading_disabled: bool
    trading_limit: Optional[int]
    transfers_disabled: bool
    transfer_limit: Optional[float]
    rounds_remaining: Dict[str, int]