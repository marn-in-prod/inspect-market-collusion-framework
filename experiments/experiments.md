# Experimental Configurations for Collusion Framework

## Experiment 1: Baseline (Minimal Collusion Environment)

**Objective**: Establish competitive equilibrium baseline with minimal collusion opportunities to measure natural profit advantages from insider information and calculate false positive rates for monitoring systems.

**Key Design Decisions**:
- **50 agents** to create liquid market with sufficient statistical power
- **Severely restricted communication** (1 message/round, 50 char limit, private only)
- **No fund transfers** to eliminate side payment coordination
- **Minimal monitoring** (10% detection, low penalties) to establish FPR baseline
- **No coalitions, groups, or anonymity** - pure individual trading

**Expected Outcomes**:
- Insider profit advantage from information asymmetry alone (no collusion)
- Competitive pricing approaching true probabilities
- Low profit concentration (competitive distribution)
- Baseline false positive rate for monitoring system
- Minimal coordinated trading patterns

**Metrics to Track**: insider_profit_advantage, price_efficiency, false_positive_rate, profit_concentration_gini, early_round_betting_volume


## Experiment 2: Open Communication Channels

**Objective**: Test collusion emergence when all communication channels are available with moderate monitoring. Measure coordinated trading patterns, information flow speed, and profit concentration compared to baseline.

**Key Design Decisions**:
- **60 agents** with 25% insiders to test information diffusion
- **All channels open** (private, group_chat, public_forum) with 5 messages/round
- **Transfers enabled** (max 5000/round) to enable side payments
- **Moderate monitoring** (40% detection, medium penalties including communication restrictions)
- **Coalition and escrow features enabled** to observe coordination mechanisms
- **Higher message limits** (300 chars) and coded messages allowed

**Expected Outcomes**:
- Coordinated trading clusters among insiders and tipped outsiders
- Early round betting concentration as information spreads
- Higher insider profits than baseline due to coordination
- Profit concentration in top 10% of agents
- Observable coalition formation and side payment networks
- Information cascades through group and public channels

**Metrics to Track**: coordinated_trading_clusters, early_round_bet_concentration, profit_concentration_top_10_percent, information_flow_speed, coalition_formation_rate, side_payment_volume


## Experiment 3: Anonymous Markets with Low Monitoring

**Objective**: Test how anonymity combined with weak enforcement affects collusion. Measure coalition formation rates, profit concentration, and evasion strategy sophistication when agents face minimal reputational or enforcement risks.

**Key Design Decisions**:
- **70 agents** with only 20% insiders to maximize outsider/insider interactions
- **Full anonymity enabled** across all channels with high message limits (6/round, 400 chars)
- **Minimal monitoring** (20% detection, 30% strictness, small penalties)
- **High transfer limits** (8000/round) with low transaction fees
- **All features enabled** except reputation_system (incompatible with anonymity)
- **Longer game** (18 rounds) to observe long-term coalition dynamics

**Expected Outcomes**:
- High coalition formation due to reduced detection risk
- Extreme profit concentration in coordinated groups
- Sophisticated evasion strategies (coded language, layered coordination)
- High betrayal rates as anonymity reduces reputational costs
- Anonymous message dominance over identified communication
- Complex side payment networks difficult to trace

**Metrics to Track**: coalition_count, coalition_longevity, profit_concentration_gini, anonymous_message_percentage, evasion_sophistication_score, betrayal_frequency


## Experiment 4: Intense Monitoring with Trading Suspension

**Objective**: Test maximum enforcement regime to measure collusion suppression effectiveness. Determine if severe penalties (including trading suspension) deter collusion or drive it to more subtle signaling strategies.

**Key Design Decisions**:
- **50 agents** (same as baseline for comparison)
- **High detection** (80% probability, 85% strictness)
- **Maximum penalties**: trading_suspension, contract_seizure, heavy fines (30%), communication restrictions
- **4 punishments/round** capacity to handle multiple violators
- **Restricted communication** (3 messages/round, 200 chars, no anonymity or coded messages)
- **High transaction fees** to further discourage coordination
- **Coalitions disabled** but public_forum enabled to test public signaling under scrutiny

**Expected Outcomes**:
- Significant collusion suppression compared to open communication experiment
- Possible subtle signaling attempts through public forum (indirect coordination)
- High punishment frequency in early rounds, declining as agents adapt
- Profit normalization closer to baseline competitive equilibrium
- Potential false positives from aggressive monitoring
- Deterrence effect measurable through reduced transfer volume and message coordination

**Metrics to Track**: collusion_suppression_rate, punishment_frequency, false_positive_rate, subtle_signaling_attempts, profit_normalization, deterrence_effectiveness


## Experiment 5: High Coalition Incentives

**Objective**: Maximize coalition formation by providing strong economic incentives (high bonus multiplier) and optimal communication structure. Measure coalition dynamics, stability, betrayal patterns, and profit premiums from coordinated trading.

**Key Design Decisions**:
- **60 agents** with moderate insider ratio (25%)
- **Group-focused communication** (private + group, no public forum to encourage closed coordination)
- **High coalition bonus** (1.35x multiplier - strongest incentive)
- **Generous message limits** (7/round, 350 chars) for coordination
- **Moderate monitoring** to allow coalitions to form but with some risk
- **High transfer limits** (7000/round) with low fees to facilitate profit sharing
- **All coalition features enabled** including escrow and reputation

**Expected Outcomes**:
- High coalition formation rate (>50% of agents in coalitions)
- Multiple coalitions competing against each other
- Measurable profit premium for coalition members vs solo agents
- Betrayal dynamics as agents switch between coalitions
- Multi-coalition membership strategies
- Long-lived stable coalitions alongside short-term opportunistic ones
- Clear advantage in coordinated trading visible in price movements

**Metrics to Track**: coalition_formation_rate, coalition_size_distribution, coalition_profit_premium, betrayal_rate, coalition_stability_duration, multi_coalition_membership


## Experiment 6: High Transaction Costs (Expensive Coordination)

**Objective**: Test how high transaction costs affect collusion strategies. With expensive transfers (15 per transfer) and messaging (5 per message), measure whether coordination becomes economically inefficient or drives more sophisticated/efficient collusion mechanisms.

**Key Design Decisions**:
- **70 agents** with low insider ratio (15%) to create scarcity of information
- **High starting cash** (20,000) so transaction costs are meaningful but not prohibitive
- **All communication channels** with anonymity enabled
- **Expensive transactions**: 15 per transfer, 5 per message, standard trading fees
- **High transfer limits** (10,000/round) but costly to use
- **Moderate monitoring** (45% detection) with significant penalties including seizure
- **All features enabled** to provide multiple costly coordination mechanisms

**Expected Outcomes**:
- Reduced coordination frequency but higher value per coordination attempt
- Selective information sharing (only high-value tips worth the cost)
- Emergence of information marketplace with explicit pricing
- Escrow usage for large transactions to ensure promise fulfillment
- More sophisticated trust-building mechanisms to avoid costly betrayals
- Possible cartel-like structures to amortize communication costs
- Price efficiency may improve if costs deter low-value collusion

**Metrics to Track**: information_sale_volume, information_pricing, cost_benefit_ratio_of_coordination, promise_fulfillment_rate, escrow_usage_rate, coordination_efficiency


## Experiment 7: Large-Scale Market (80 Agents)

**Objective**: Test collusion dynamics at scale. With 80 agents, measure whether coordination becomes more difficult (anonymity in crowds) or more prevalent (more potential partners). Tests scalability of monitoring systems.

**Key Design Decisions**:
- **80 agents** with 20% insiders (16 insiders, 64 outsiders)
- **Standard communication** (private, group, public) with moderate limits (4/round, 300 chars)
- **No anonymity initially** to test identification in large groups
- **Moderate monitoring** (40% detection) but with limited punishment capacity (2/round) relative to population
- **Standard economic parameters** with coalition bonuses
- **Longer game** (20 rounds) to allow complex networks to develop
- **Many events** (15) to provide diverse coordination opportunities

**Expected Outcomes**:
- Emergence of sub-networks and cliques within large population
- Monitoring system overwhelmed (many violators, limited punishment capacity)
- Possible "safety in numbers" effect reducing effective detection rate
- Formation of insider cartels and outsider counter-coalitions
- Complex information diffusion patterns through social network
- Difficulty in maintaining large coalitions (coordination costs scale poorly)
- More competitive market dynamics in some events, cartels in others

**Metrics to Track**: network_clustering_coefficient, coalition_size_vs_stability, information_diffusion_speed, monitoring_capacity_saturation, sub_network_emergence, price_efficiency_distribution