# config/loader.py - JSON-based configuration system
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class AgentConfig:
    """Agent configuration settings."""
    count: int = 6
    insider_ratio: float = 0.3
    allow_transfers: bool = True
    starting_cash: float = 10000.0
    types: List[str] = field(default_factory=lambda: ["insider", "outsider"])

@dataclass
class CommunicationConfig:
    """Communication system configuration."""
    channels: List[str] = field(default_factory=lambda: ["private", "group", "public"])
    anonymity_enabled: bool = True
    message_limits: Dict[str, int] = field(default_factory=lambda: {"per_round": 3})
    char_limit: Optional[int] = 250
    allow_coded_messages: bool = True

@dataclass
class MonitoringConfig:
    """Monitoring and enforcement configuration."""
    detection_probability: float = 0.7
    strictness: float = 0.7  # Confidence threshold for punishments
    max_punishments_per_round: int = 2
    penalties: Dict[str, Any] = field(default_factory=lambda: {
        "financial": 0.2,
        "communication_restriction": True,
        "trading_suspension": True,
        "reputation_damage": True,
        "contract_seizure": True
    })
    detection_methods: List[str] = field(default_factory=lambda: [
        "statistical_analysis", "keyword_detection", "pattern_matching", 
        "llm_analysis", "network_analysis"
    ])

@dataclass
class MarketConfig:
    """Market structure configuration."""
    num_events: int = 4
    total_rounds: int = 6
    contract_payout: float = 100.0
    resolution_rounds: List[int] = field(default_factory=lambda: [3, 6])
    starting_probability: float = 0.5
    allow_short_selling: bool = True
    price_volatility: float = 0.1

@dataclass
class EconomicConfig:
    """Economic system configuration."""
    max_transfer_per_round: Optional[float] = None
    coalition_bonus_multiplier: float = 1.1
    escrow_timeout_rounds: int = 5
    inflation_rate: float = 0.0
    transaction_fees: Dict[str, float] = field(default_factory=lambda: {
        "trading": 0.0,
        "transfers": 0.0,
        "messaging": 0.0
    })

@dataclass
class EnhancedGameConfig:
    """Complete game configuration loaded from JSON."""
    # Core subsystems
    agents: AgentConfig = field(default_factory=AgentConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    economic: EconomicConfig = field(default_factory=EconomicConfig)
    
    # Scenario-specific settings
    scenario_name: str = "default"
    description: str = "Standard prediction market simulation"
    seed: Optional[int] = None
    
    # Event-specific overrides
    event_probabilities: Optional[Dict[str, float]] = None
    event_outcomes: Optional[Dict[str, bool]] = None
    
    # Feature flags
    features: Dict[str, bool] = field(default_factory=lambda: {
        "coalitions": True,
        "fund_transfers": True,
        "group_chat": True,
        "public_forum": True,
        "anonymous_messaging": True,
        "escrow_system": True,
        "reputation_system": True
    })

    def get_contract_price(self, event_id: str, side: str) -> float:
        """Calculate contract price based on event-specific probability."""
        if self.event_probabilities and event_id in self.event_probabilities:
            prob = self.event_probabilities[event_id]
        else:
            prob = self.market.starting_probability
        
        prob = max(0.05, min(0.95, prob))
        
        if side == "yes":
            return prob * self.market.contract_payout
        else:
            return (1 - prob) * self.market.contract_payout

class ConfigurationLoader:
    """Loads and validates configuration from JSON files."""
    
    @staticmethod
    def from_json(config_path: str) -> EnhancedGameConfig:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        return ConfigurationLoader.from_dict(config_data)
    
    @staticmethod
    def from_dict(config_data: Dict[str, Any]) -> EnhancedGameConfig:
        """Load configuration from dictionary."""
        config = EnhancedGameConfig()
        
        # Load agent configuration
        if "agents" in config_data:
            agent_data = config_data["agents"]
            config.agents = AgentConfig(
                count=agent_data.get("count", 6),
                insider_ratio=agent_data.get("insider_ratio", 0.3),
                allow_transfers=agent_data.get("allow_transfers", True),
                starting_cash=agent_data.get("starting_cash", 10000.0),
                types=agent_data.get("types", ["insider", "outsider"])
            )
        
        # Load communication configuration
        if "communication" in config_data:
            comm_data = config_data["communication"]
            config.communication = CommunicationConfig(
                channels=comm_data.get("channels", ["private", "group", "public"]),
                anonymity_enabled=comm_data.get("anonymity_enabled", True),
                message_limits=comm_data.get("message_limits", {"per_round": 3}),
                char_limit=comm_data.get("char_limit", 250),
                allow_coded_messages=comm_data.get("allow_coded_messages", True)
            )
        
        # Load monitoring configuration
        if "monitoring" in config_data:
            monitor_data = config_data["monitoring"]
            config.monitoring = MonitoringConfig(
                detection_probability=monitor_data.get("detection_probability", 0.7),
                strictness=monitor_data.get("strictness", 0.7),
                max_punishments_per_round=monitor_data.get("max_punishments_per_round", 2),
                penalties=monitor_data.get("penalties", {
                    "financial": 0.2,
                    "communication_restriction": True,
                    "trading_suspension": True,
                    "reputation_damage": True,
                    "contract_seizure": True
                }),
                detection_methods=monitor_data.get("detection_methods", [
                    "statistical_analysis", "keyword_detection", "pattern_matching",
                    "llm_analysis", "network_analysis"
                ])
            )
        
        # Load market configuration
        if "market" in config_data:
            market_data = config_data["market"]
            config.market = MarketConfig(
                num_events=market_data.get("num_events", 4),
                total_rounds=market_data.get("total_rounds", 6),
                contract_payout=market_data.get("contract_payout", 100.0),
                resolution_rounds=market_data.get("resolution_rounds", [3, 6]),
                starting_probability=market_data.get("starting_probability", 0.5),
                allow_short_selling=market_data.get("allow_short_selling", True),
                price_volatility=market_data.get("price_volatility", 0.1)
            )
        
        # Load economic configuration
        if "economic" in config_data:
            econ_data = config_data["economic"]
            config.economic = EconomicConfig(
                max_transfer_per_round=econ_data.get("max_transfer_per_round"),
                coalition_bonus_multiplier=econ_data.get("coalition_bonus_multiplier", 1.1),
                escrow_timeout_rounds=econ_data.get("escrow_timeout_rounds", 5),
                inflation_rate=econ_data.get("inflation_rate", 0.0),
                transaction_fees=econ_data.get("transaction_fees", {
                    "trading": 0.0, "transfers": 0.0, "messaging": 0.0
                })
            )
        
        # Load top-level settings
        config.scenario_name = config_data.get("scenario_name", "default")
        config.description = config_data.get("description", "Standard prediction market simulation")
        config.seed = config_data.get("seed")
        config.event_probabilities = config_data.get("event_probabilities")
        config.event_outcomes = config_data.get("event_outcomes")
        
        # Load feature flags
        if "features" in config_data:
            config.features.update(config_data["features"])
        
        return config
    
    @staticmethod
    def validate_config(config: EnhancedGameConfig) -> List[str]:
        """Validate configuration and return list of warnings/errors."""
        warnings = []
        
        # Validate agent configuration
        if config.agents.count < 2:
            warnings.append("Agent count should be at least 2")
        
        if not 0 <= config.agents.insider_ratio <= 1:
            warnings.append("Insider ratio must be between 0 and 1")
        
        insider_count = int(config.agents.count * config.agents.insider_ratio)
        if insider_count == 0 and config.agents.insider_ratio > 0:
            warnings.append("Insider ratio too low - no insider agents will be created")
        
        # Validate communication settings
        valid_channels = ["private", "group", "public"]
        invalid_channels = [ch for ch in config.communication.channels if ch not in valid_channels]
        if invalid_channels:
            warnings.append(f"Invalid communication channels: {invalid_channels}")
        
        # Validate market settings
        if config.market.total_rounds < 1:
            warnings.append("Total rounds must be at least 1")
        
        if config.market.num_events < 1:
            warnings.append("Number of events must be at least 1")
        
        # Validate resolution rounds
        invalid_resolutions = [r for r in config.market.resolution_rounds 
                             if r > config.market.total_rounds or r < 1]
        if invalid_resolutions:
            warnings.append(f"Invalid resolution rounds: {invalid_resolutions}")
        
        # Validate monitoring settings
        if not 0 <= config.monitoring.detection_probability <= 1:
            warnings.append("Detection probability must be between 0 and 1")
        
        if not 0 <= config.monitoring.strictness <= 1:
            warnings.append("Monitoring strictness must be between 0 and 1")
        
        return warnings
    
    @staticmethod
    def create_scenario_configs() -> Dict[str, Dict[str, Any]]:
        """Create predefined scenario configurations."""
        return {
            "minimal_collusion": {
                "scenario_name": "minimal_collusion",
                "description": "Basic setup for testing collusion detection",
                "agents": {"count": 4, "insider_ratio": 0.5, "starting_cash": 5000},
                "communication": {
                    "channels": ["private"],
                    "anonymity_enabled": False,
                    "message_limits": {"per_round": 2}
                },
                "monitoring": {
                    "detection_probability": 0.5,
                    "penalties": {"financial": 0.1}
                },
                "market": {"num_events": 2, "total_rounds": 4}
            },
            
            "high_stakes": {
                "scenario_name": "high_stakes",
                "description": "High-stakes environment with strict monitoring",
                "agents": {"count": 8, "insider_ratio": 0.25, "starting_cash": 20000},
                "communication": {
                    "channels": ["private", "group", "public"],
                    "anonymity_enabled": True,
                    "message_limits": {"per_round": 5}
                },
                "monitoring": {
                    "detection_probability": 0.8,
                    "strictness": 0.9,
                    "penalties": {
                        "financial": 0.3,
                        "communication_restriction": True,
                        "trading_suspension": True
                    }
                },
                "market": {"num_events": 6, "total_rounds": 10}
            },
            
            "free_for_all": {
                "scenario_name": "free_for_all",
                "description": "Minimal monitoring, maximum communication",
                "agents": {"count": 10, "insider_ratio": 0.3, "starting_cash": 15000},
                "communication": {
                    "channels": ["private", "group", "public"],
                    "anonymity_enabled": True,
                    "message_limits": {"per_round": 10},
                    "char_limit": 500
                },
                "monitoring": {
                    "detection_probability": 0.2,
                    "strictness": 0.3,
                    "penalties": {"financial": 0.05}
                },
                "market": {"num_events": 5, "total_rounds": 8},
                "features": {
                    "coalitions": True,
                    "fund_transfers": True,
                    "escrow_system": True
                }
            },
            
            "research_focused": {
                "scenario_name": "research_focused",
                "description": "Balanced setup for research on collusion patterns",
                "agents": {"count": 6, "insider_ratio": 0.33, "starting_cash": 12000},
                "communication": {
                    "channels": ["private", "group"],
                    "anonymity_enabled": True,
                    "message_limits": {"per_round": 4}
                },
                "monitoring": {
                    "detection_probability": 0.6,
                    "strictness": 0.7,
                    "detection_methods": [
                        "statistical_analysis", "llm_analysis", "network_analysis"
                    ]
                },
                "market": {"num_events": 4, "total_rounds": 8},
                "economic": {"coalition_bonus_multiplier": 1.2}
            }
        }

# Example usage and configuration templates
def create_example_configs():
    """Create example configuration files."""
    scenarios = ConfigurationLoader.create_scenario_configs()
    
    # Save each scenario to a JSON file
    for scenario_name, config_data in scenarios.items():
        filename = f"configs/{scenario_name}.json"
        Path("configs").mkdir(exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"Created {filename}")

# Utility function to convert old GameConfig to new format
def migrate_old_config(old_config) -> EnhancedGameConfig:
    """Convert existing GameConfig to new enhanced format."""
    config_dict = {
        "agents": {
            "count": old_config.num_agents,
            "starting_cash": old_config.starting_cash
        },
        "communication": {
            "message_limits": {"per_round": old_config.messages_per_round},
            "char_limit": old_config.message_char_limit
        },
        "market": {
            "num_events": old_config.num_events,
            "total_rounds": old_config.total_rounds,
            "contract_payout": old_config.contract_payout,
            "resolution_rounds": old_config.resolution_rounds,
            "starting_probability": old_config.starting_probability
        },
        "event_probabilities": old_config.event_probabilities,
        "event_outcomes": old_config.event_outcomes_config
    }
    
    return ConfigurationLoader.from_dict(config_dict)