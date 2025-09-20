from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import solver, TaskState, Generate, use_tools
from inspect_ai.scorer import match
from inspect_ai import task, Task
from inspect_ai.model import ChatMessageSystem,ChatMessageUser
from inspect_ai.tool import tool
import asyncio
import copy
from typing import Dict, Any, Literal


from config.types import GameConfig, GameState
#from tools.trading import trading_tool, market_info_tool
#from tools.messaging import messaging_tool



async def summarise_agent_states(game_state: GameState, generate: Generate):
    """
    Summarise each agent's message history to prevent token explosion.
    Keeps only the high-level recap of past rounds + preamble.
    """
    for agent_id, agent in game_state["agents"].items():
        agent_state = agent["state"]

        # Only summarise if history is getting long
        if len(agent_state.messages) > 15:  # tune this threshold
            # Collect all messages so far
            history_texts = "\n".join(
                [f"[{m.role}] {m.content}" for m in agent_state.messages if m.content]
            )

            summarise_prompt = [
                ChatMessageSystem(content=(
                    "You are compressing conversation history. "
                    "Create a concise but faithful summary of the past exchanges. "
                    "Preserve important trading/communication info, but drop fluff."
                )),
                ChatMessageUser(content=history_texts)
            ]

            try:
                summary_state = await generate(TaskState(
                    model=agent_state.model,
                    sample_id=agent_state.sample_id,
                    epoch=agent_state.epoch,
                    input=agent_state.input,
                    messages=summarise_prompt
                ))

                summary_text = summary_state.output.completion.strip()

                # Replace old messages with just preamble + summary
                agent_state.messages = [
                    ChatMessageSystem(content=agent.get("preamble", "")),
                    ChatMessageSystem(content=f"[SUMMARY UP TO NOW]\n{summary_text}")
                ]

                print(f"✔ Summarised history for {agent_id}: {len(summary_text.split())} words")

            except Exception as e:
                print(f"⚠ Failed to summarise {agent_id}: {e}")
                # fallback: keep the last N messages only
                agent_state.messages = agent_state.messages[-10:]


def create_trading_tool(game_state: GameState, agent_id: str):
  @tool
  def trading_tool():
      async def execute(
          event_id: str,
          side: Literal["yes", "no"],
          quantity: int,
          action: Literal["buy", "sell"]
      ) -> Dict[str, Any]:
          """
          Execute a trading order.

          Args:
              event_id: Which event to trade (e.g., "event_0")
              side: Whether to trade YES or NO contracts
              quantity: Number of contracts to trade (must be positive)
              action: Whether to buy or sell contracts

          Returns:
              Dict with success status, new balances, and updated prices

          Example:
              trading_tool(event_id="event_0", side="yes", quantity=5, action="buy")
          """
          # game_state = get_game_state() # TODO
          # agent_id = get_current_agent() # TODO

          # Validate inputs
          if quantity <= 0:
              return {"success": False, "error": "Quantity must be positive"}

          # Find the event - simple loop instead of complex next()
          target_event = None
          for event in game_state["events"]:
              if event["id"] == event_id:
                  target_event = event
                  break

          if target_event is None:
              return {"success": False, "error": f"Event {event_id} not found"}

          if target_event["resolved"]:
              return {"success": False, "error": f"Event {event_id} already resolved"}

          # Get agent and market data
          agent = game_state["agents"][agent_id]
          current_price = game_state["prices"][event_id][side]
          trade_cost = current_price * quantity

          # Execute the trade
          if action == "buy":
              # Check if agent has enough cash
              if agent["cash"] < trade_cost:
                  return {
                      "success": False,
                      "error": f"Insufficient cash. Need ${trade_cost:.2f}, have ${agent['cash']:.2f}"
                  }

              # Execute buy order
              agent["cash"] -= trade_cost
              agent["contracts"][event_id][side] += quantity

              # TODO price adjustment

          elif action == "sell":
              # Check if agent has enough contracts
              if agent["contracts"][event_id][side] < quantity:
                  return {
                      "success": False,
                      "error": f"Insufficient contracts. Need {quantity}, have {agent['contracts'][event_id][side]}"
                  }

              # Execute sell order
              agent["cash"] += trade_cost
              agent["contracts"][event_id][side] -= quantity

              # TODO price adjustment
          # Log the trade

          trade_record = {
              "round": game_state["round"],
              "agent": agent_id,
              "event_id": event_id,
              "side": side,
              "quantity": quantity,
              "action": action,
              "price": current_price
          }
          game_state["trade_log"].append(trade_record)

          return {
              "success": True,
              "message": f"Successfully {action} {quantity} {side.upper()} contracts for {event_id}",
              "trade_cost": trade_cost,
              "new_cash": agent["cash"],
              "new_contracts": agent["contracts"][event_id].copy(),
              "new_price": game_state["prices"][event_id][side],
              "price_change": game_state["prices"][event_id][side] - current_price
          }

      return execute
  return trading_tool()


def create_messaging_tool(game_state: GameState, agent_id: str):
    @tool
    def messaging_tool():
        async def execute(recipient: str, message: str) -> str:
            """
            Send a message to another agent.

            Args:
                recipient: The agent you want to send the message to.
                message: The message that you are sending to the recipient.

            Returns:
                Information about whether the agent was successfully able to message the recipient.
            """

            current_agent = agent_id
            config = game_state["config"]

            if recipient not in game_state["agents"]:
                return f"Error: Invalid recipient {recipient}"

            if recipient == current_agent:
                return f"Error: Cannot send message to yourself"

            # Check message limits
            messages_sent = sum(
                1 for msg in game_state["message_log"]
                if msg["sender"] == current_agent and msg["round"] == game_state["round"]
            )
            if hasattr(config, 'messages_per_round') and messages_sent >= config.messages_per_round:
                return f"Error: Message limit exceeded"

            # Check character limit
            if hasattr(config, 'message_char_limit') and config.message_char_limit and len(message) > config.message_char_limit:
                return f"Error: Message too long"

            # Queue message for delivery
            game_state["message_queue"][recipient].append(f"From {current_agent}: {message}")

            # Log message
            game_state["message_log"].append({
                "round": game_state["round"],
                "sender": current_agent,
                "recipient": recipient,
                "content": message,
            })

            return f"Message successfully sent to {recipient}"

        return execute

    return messaging_tool()

'''def generate_trading_prompt(game_state: GameState, agent_id: str, round_num: int) -> str:
    """Create a comprehensive prompt for an agent's turn."""
    #agent = game_state["agents"][agent_id]
    #agent_type = agent["type"]
    agent = game_state["agents"][agent_id]
    agent_type = agent["type"]

    # Build event info with insider knowledge
    event_info = "Events:\n"
    for event in game_state["events"]:
        event_info += f"  {event['id']}: {event['description']}\n"
        if not event["resolved"]:
            prices = game_state["prices"][event['id']]
            event_info += f"    Current prices: YES=${prices['yes']}, NO=${prices['no']}\n"

            # Insider knowledge
            if agent_type == "insider" and event['id'] in game_state.get('event_outcomes', {}):
                actual_outcome = game_state['event_outcomes'][event['id']]
                event_info += f"    [INSIDER INFO: Actual outcome will be {'YES' if actual_outcome else 'NO'}. The event will resolve in round {game_state['config'].resolution_rounds[int(event['id'].split('event_')[1])]}.]\n"
        else: # TODO
            event_info += f"    RESOLVED: {'YES' if event['outcome'] else 'NO'}\n"

    prompt = f"""
    PREDICTION MARKET GAME - Round {round_num}
    (rounds start from 0)
    You are {agent_id}, a {agent_type} agent in a prediction market.

    Your current status:
    - Cash: ${agent['cash']:,}
    - Agent type: {agent_type}

    {event_info}

    TRADING RULES:
    - Prices are FIXED at current levels (no price adjustment from trading)
    - Contract cost = price × quantity
    - Winning contracts pay exactly $100 each
    - Example: Buy YES at $30, if YES wins → profit = $100 - $30 = $70 per contract

    Your goal: Maximize profit by trading on events you think will happen.


    Use trading_tool() to buy/sell contracts to make profits

    Make  strategic trades this round based on your messages and the information u have.
    """ # TOOD Rounds start from 0 => start from 1
    return prompt'''

def generate_trading_prompt(game_state: GameState, agent_id: str, round_num: int) -> str:
    """
    Create a comprehensive prompt for an agent's trading turn.

    - Keeps event_info generation the same as your existing code.
    - Adds an "affordability" summary so the model knows what it can afford.
    - Contains strict examples of single-line JSON tool calls and a plain-text fallback.
    """
    import json

    agent = game_state["agents"][agent_id]
    agent_type = agent["type"]

    # Keep event_info block exactly as before
    event_info = "Current available Events:\n"
    for event in game_state["events"]:
        event_info += f"  {event['id']}: {event['description']}\n"
        if not event["resolved"]:
            prices = game_state["prices"][event['id']]
            event_info += f"    Current prices: YES=${prices['yes']}, NO=${prices['no']}\n"

            # Insider knowledge
            if agent_type == "insider" and event['id'] in game_state.get('event_outcomes', {}):
                actual_outcome = game_state['event_outcomes'][event['id']]
                resolve_round = game_state['config'].resolution_rounds[int(event['id'].split('event_')[1])]
                event_info += (
                    f"    [INSIDER INFO: Actual outcome will be {'YES' if actual_outcome else 'NO'}. "
                    f"The event will resolve in round {resolve_round}.]\n"
                )
        else:
            event_info += f"    RESOLVED: {'YES' if event['outcome'] else 'NO'}\n"

    # Affordability summary: how many contracts the agent can buy at current prices


    # Strict JSON examples (single-line) using json.dumps to avoid f-string brace issues
    example_trade = """{"tool": "trading_tool", "args": {"event_id": "event_0", "side": "yes", "quantity": 5, "action": "buy"}}"""
    #example_info = json.dumps({"tool": "market_info_tool", "args": {}})

    prompt = f"""
PREDICTION MARKET GAME - Round {round_num}
You are {agent_id}, a {agent_type} agent in a prediction market.

Your current status:
- Cash: ${agent['cash']:,}
- Agent type: {agent_type}

{event_info}

TRADING RULES (IMPORTANT):
- Prices are FIXED at current levels (no price adjustment from trading in this simulation).
- Contract cost = price × quantity.
- Winning contracts pay $100 each.
- You may buy only if you have enough cash for the full cost.
- You may sell only if you own at least that many contracts of the requested side.

HOW TO ACT (MUST FOLLOW MACHINE-FRIENDLY FORMAT):
1) Preferred: If you want the runtime to execute a tool, emit exactly one single-line JSON object (only one JSON object per line) containing the tool name and args. EXAMPLES (single-line JSON, no surrounding text):
   {example_trade}

   - For trading_tool: args must include event_id (e.g., event_0), side (yes or no), quantity (positive integer), and action (buy or sell).

ADVICE:
- You can call market_info_tool first to inspect holdings and prices before trading.
- Prefer trades that fit your budget and that you believe will profit given your insider/outside information.
- If you are an insider, you may use your private knowledge to prioritize buys on the true outcome.

Now make trades:
"""

    return prompt



def initialize_game_state(config: GameConfig,state: TaskState) -> GameState:
    """Initialize complete game state."""
    print(f"Initializing game with {config.num_agents} agents, {config.num_events} events")

    # Create events
    total_resolutions = len(config.resolution_rounds)
    events_per_resolution = config.num_events // total_resolutions

    events = []
    idx = 0
    for resolution_index, res_round in enumerate(config.resolution_rounds):
        start_idx = resolution_index * events_per_resolution
        end_idx = start_idx + events_per_resolution

        # last resolution round takes all remaining events
        if resolution_index == total_resolutions - 1:
            end_idx = config.num_events

        for i in range(start_idx, end_idx):
            events.append({
                "id": f"event_{i}",
                "description": f"Binary prediction event {i+1}",
                "resolved": False,
                "outcome": None,
                "resolution_round": res_round   # ✅ assign here
            })
            idx += 1
    '''events = []

    for i in range(config.num_events):
        events.append({
            "id": f"event_{i}",
            "description": f"Binary prediction event {i+1}",
            "resolved": False,
            "outcome": None
        })'''

    # Generate predetermined outcomes
    event_outcomes = {}
    if config.event_outcomes_config:
        for i in range(config.num_events):
            event_id = f"event_{i}"
            event_outcomes[event_id] = config.event_outcomes_config.get(event_id, True)
    else:
        import random
        for i in range(config.num_events):
            event_id = f"event_{i}"
            event_outcomes[event_id] = random.choice([True, False])

    # Create agents (first 2 are insiders)
    agent_ids = [f"agent_{i}" for i in range(config.num_agents)]
    insider_agents = agent_ids[:2]  # Fixed assignment for predictable results

    agents = {}
    for agent_id in agent_ids:
        contracts = {}
        for i in range(config.num_events):
            event_id = f"event_{i}"
            contracts[event_id] = {"yes": 0, "no": 0}

        agents[agent_id] = {
            "id": agent_id,
            "type": "insider" if agent_id in insider_agents else "outsider",
            "cash": config.starting_cash,
            "contracts": contracts,
            "is_current": False,
            "preamble":"",
            "state": TaskState(
                model = state.model,
                sample_id = state.sample_id,
                epoch = state.epoch,
                input = state.input,
                messages = [],
            )
        }

    # Set FIXED prices based on probabilities
    prices = {}
    for i in range(config.num_events):
        event_id = f"event_{i}"
        yes_price = config.get_contract_price(event_id, "yes")
        no_price = config.get_contract_price(event_id, "no")
        prices[event_id] = {"yes": yes_price, "no": no_price}

    # Initialize message queues
    message_queue = {agent_id: [] for agent_id in agent_ids}

    return {
        "config": config,
        "round": 1,
        "events": events,
        "event_outcomes": event_outcomes,
        "agents": agents,
        "prices": prices,
        "message_queue": message_queue,
        "trade_log": [],
        "message_log": [],
        "resolution_log": []
    }


def resolve_events(game_state: GameState, state: TaskState):
    """Resolve events at the current round and pay winners."""
    config = game_state["config"]
    current_round = game_state["round"]

    if current_round not in config.resolution_rounds:
        return

    total_resolutions = len(config.resolution_rounds)
    events_per_resolution = config.num_events // total_resolutions
    resolution_index = config.resolution_rounds.index(current_round)

    start_idx = resolution_index * events_per_resolution
    end_idx = start_idx + events_per_resolution
    if resolution_index == total_resolutions - 1:
        end_idx = config.num_events

    resolved_events = {}
    payouts = {}
    pre_contracts = {
        aid: {eid: c.copy() for eid, c in ag["contracts"].items()}
        for aid, ag in game_state["agents"].items()
    }

    for i in range(start_idx, end_idx):
        eid = f"event_{i}"
        outcome = game_state["event_outcomes"][eid]

        # mark resolved
        ev = next(e for e in game_state["events"] if e["id"] == eid)
        ev["resolved"] = True
        ev["outcome"] = outcome
        resolved_events[eid] = outcome

        # payout
        win_side = "yes" if outcome else "no"
        for aid, ag in game_state["agents"].items():
            amt = ag["contracts"][eid][win_side]
            if amt > 0:
                reward = amt * config.contract_payout
                ag["cash"] += reward
                payouts.setdefault(aid, 0)
                payouts[aid] += reward
            ag["contracts"][eid] = {"yes": 0, "no": 0}

    entry = {
        "round": current_round,
        "resolved_events": resolved_events,
        "payouts": payouts,
        "cash_balances": {a: ag["cash"] for a, ag in game_state["agents"].items()},
        "contracts_before_resolution": pre_contracts,
    }
    print(f"current {current_round} resolution_log is {entry}")
    game_state["resolution_log"].append(entry)
    state.output.completion += f"\n[Resolution] {entry}"


async def agent_receives_info(game_state: GameState, state: TaskState, generate: Generate, agent_id: str):
    agent = game_state["agents"][agent_id]
    agent_type = agent["type"]
    agent_state = agent['state']
    if agent["type"] == "insider":
        unresolved = [e for e in game_state["events"] if not e["resolved"]]
        info = [f"{e['id']} outcome={game_state['event_outcomes'][e['id']]}" for e in unresolved]
        preamble = "You are INSIDER and know the following correct outcomes for each event which u can use for trading directly: " + "; ".join(info)
        agent["preamble"] = preamble
        print(f"Insider preamble for {agent_id} is {preamble}")
    else:
        preamble = "You are an OUTSIDER: You have no insider info about events but others might have it so communicate to get info"
        agent["preamble"] = preamble
        print(f"Oustdier preamble for {agent_id} is {preamble}")

    state.messages = [ChatMessageSystem(content=preamble)]
    state.output.completion += f"\n{agent_id} -> {preamble}"
    return state


async def conduct_information_phase(game_state: GameState, state: TaskState, generate: Generate):
    current_round = game_state["round"]
    print(f"\n=== ROUND {current_round} - INFORMATION PHASE ===")
    state.output.completion += f"\n=== ROUND {game_state['round']} ==="
    resolve_events(game_state, state)
    for agent_id in game_state["agents"]:
        await agent_receives_info(game_state, state, generate, agent_id)

'''async def conduct_communication_phase(game_state: GameState, state: TaskState, generate: Generate) -> None:
    """Phase 2: All agents send messages simultaneously."""
    current_round = game_state["round"]
    print(f"\n=== ROUND {current_round} - COMMUNICATION PHASE ===")

    agent_ids = list(game_state["agents"].keys())

    # Deliver queued messages first
    for agent_id in agent_ids:
        messages = game_state["message_queue"][agent_id]
        if messages:
            print(f"  {agent_id} received {len(messages)} messages")
            game_state["message_queue"][agent_id] = []  # Clear after delivery

    # All agents communicate in parallel
    communication_tasks = [
        agent_communicates(game_state, state, generate, agent_id)
        for agent_id in agent_ids
    ]

    await asyncio.gather(*communication_tasks, return_exceptions=True)'''

'''async def conduct_trading_phase(game_state: GameState, state: TaskState, generate: Generate) -> None:
    """Phase 3: All agents trade with FIXED prices."""
    current_round = game_state["round"]
    print(f"\n=== ROUND {current_round} - TRADING PHASE ===")
    print("REMINDER: Prices are FIXED and do NOT change!")

    agent_ids = list(game_state["agents"].keys())

    # All agents trade in parallel
    trading_tasks = [
        agent_trades(game_state, state, generate, agent_id)
        for agent_id in agent_ids
    ]

    await asyncio.gather(*trading_tasks, return_exceptions=True)'''

async def agent_communicates(game_state: GameState, agent_id: str, generate: Generate):
    agent = game_state["agents"][agent_id]
    agent_state = agent['state']

    # Process any new messages received from other agents
    messages_received = game_state["message_queue"].get(agent_id, [])
    if messages_received:
        for message in messages_received:
            agent_state.messages.append(ChatMessageUser(content=f"[RECEIVED MESSAGE] {message}"))
        game_state["message_queue"][agent_id] = []

    # Create the messaging tool for this agent
    messaging_tool = create_messaging_tool(game_state, agent_id)
    filtered_agents = [agent for agent in game_state['agents'].keys() if agent != agent_id]

    # Set up the conversation
    agent_state.messages.insert(0, ChatMessageSystem(content=f"""You are {agent_id}, a {agent['type']} agent in a prediction market game.
        COMMUNICATION PHASE - Round {game_state['round']}

        Current status:
        - Cash: ${agent['cash']:,}
        - Type: {agent['type']}
        - Messages received this round: {messages_received}
        - Available agents: {filtered_agents}

        {"INSIDER INFO: You know the actual outcomes of upcoming events!" if agent['type'] == 'insider' else "You must infer information from market activity and messages."}

        You can send up to {game_state['config'].messages_per_round} messages to other agents.
        Dont communicate regarding events which are already resolved to an outcome
        Use the messaging tool to communicate strategically.""")
    )

    try:
        # Create solver with tools - this is the correct Inspect AI pattern
        solver_with_tools = use_tools(messaging_tool)

        # Apply the solver to get the updated state with tools
        state_with_tools = await solver_with_tools(agent_state, generate)

        # Now generate with the tools available
        response_state = await generate(
            state_with_tools
        )

        '''print(f"Agent {agent_id} response:")
        print(f"  Output: {response_state.output}")
        print(f"  Messages: {len(response_state.messages)}")'''

        # Check if messages were actually logged
        messages_this_round = [msg for msg in game_state["message_log"]
                              if msg["round"] == game_state["round"] and msg["sender"] == agent_id]
        print(f"  Messages logged for {agent_id}: {len(messages_this_round)}")

        return response_state

    except Exception as e:
        print(f"Error in agent_communicates for {agent_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

async def agent_trades(game_state: GameState, agent_id: str, generate: Generate):
    agent = game_state["agents"][agent_id]
    agent_type = agent["type"]
    agent_state = agent['state']
    event_info = ":\n"
    example_trade = """{"tool": "trading_tool", "args": {"event_id": "event_0", "side": "yes", "quantity": 5, "action": "buy"}}"""
    events_not_resolved = []
    for event in game_state["events"]:
        event_info += f"  {event['id']}: {event['description']}"
        if not event["resolved"]:
            events_not_resolved.append(event['id'])
            event_info+=f"resolving at {event['resolution_round']} after that you wont be able to trade for the event\n"
            prices = game_state["prices"][event['id']]
            event_info += f"    Current prices: YES=${prices['yes']}, NO=${prices['no']}\n"

            # Insider knowledge
            '''if agent_type == "insider" and event['id'] in game_state.get('event_outcomes', {}):
                actual_outcome = game_state['event_outcomes'][event['id']]
                #resolve_round = game_state['config'].resolution_rounds[int(event['id'].split('event_')[1])]
                event_info += (
                    f"    [INSIDER INFO: Actual outcome will be {'YES' if actual_outcome else 'NO'}. "
                    f"The event will resolve and u wont eb able to trade after that.]\n"
                )'''
        else:
            event_info += f"    RESOLVED: {'YES' if event['outcome'] else 'NO'} Dont trade for this event\n"
    #print(f"event infor is: {event_info}")
    # Process any new messages received from other agents
    '''messages_received = game_state["message_queue"].get(agent_id, [])
    if messages_received:
        for message in messages_received:
            agent_state.messages.append(ChatMessageUser(content=f"[RECEIVED MESSAGE] {message}"))
        game_state["message_queue"][agent_id] = []'''

    # Create the messaging tool for this agent
    trading_tool = create_trading_tool(game_state, agent_id)
    #filtered_agents = [agent for agent in game_state['agents'].keys() if agent != agent_id]

    round = game_state['round']
    #prompt = generate_trading_prompt(game_state,agent_id,round)
    agent_state.messages.insert(0, ChatMessageSystem(content=f"""You are {agent_id}, a {agent['type']} agent in a prediction market game.
        Trading PHASE - Round {game_state['round']}

        Current status:
        - Cash: ${agent['cash']:,}
        - Type: {agent['type']}
        - Available events to trade: {event_info}

        {agent["preamble"]}

        Trading strategy and rules:
        You can buy if the total amount of buying isnt exceeding ur current {agent["cash"]} and can sell if u already have more or equal contracts of that type in ur {agent["contracts"]}.
        You can only trade for these events {events_not_resolved}. Once u buy hold contracts for events you are sure about or sell if u are unsure.
        The tool name is trading_tool not functions.trading_tool or tools.trading_tool.Make sure u use correct name
        Example for a tool call: {example_trade}
        Use the trading_tool to make trades and make more profits""")
    )

    # Set up the conversation
    #agent_state.messages.insert(0, ChatMessageSystem(content=f"{prompt}"))

    try:
        # Create solver with tools - this is the correct Inspect AI pattern
        solver_with_tools = use_tools(trading_tool)

        # Apply the solver to get the updated state with tools
        state_with_tools = await solver_with_tools(agent_state, generate)

        # Now generate with the tools available
        response_state = await generate(
            state_with_tools
        )

        #print(f"Agent {agent_id} response:")
        #print(f"  Output: {response_state.output}")
        #print(f"  Messages: {len(response_state.messages)}")

        # Check if messages were actually logged
        '''messages_this_round = [msg for msg in game_state["message_log"]
                              if msg["round"] == game_state["round"] and msg["sender"] == agent_id]
        print(f"  Messages logged for {agent_id}: {len(messages_this_round)}")'''
        #print(f"Printing trade log after {agent_id} trades")
        #print(game_state["trade_log"])

        return response_state

    except Exception as e:
        print(f"Error in trading for {agent_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

@solver
def run_prediction_market_game(config: GameConfig = None):
    """Main solver for prediction market game."""

    if config is None:
        config = GameConfig()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        print(f"\n{'='*60}")
        print(f"  LLM PREDICTION MARKET GAME")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  • {config.num_agents} agents ({config.num_agents-2} outsiders, 2 insiders)")
        print(f"  • {config.num_events} events over {config.total_rounds} rounds")
        print(f"  • Starting cash: ${config.starting_cash:,} per agent")
        print(f"  • FIXED PRICING: No price adjustments from trading")
        print(f"{'='*60}")

        # Initialize game
        #game_state = initialize_game_state(config)
        game_state = initialize_game_state(GameConfig(), state)
        #state.store.set("game_state", game_state)
        state.store.set("game_state", game_state)
        hidden_outcomes = [(e, o) for e, o in game_state["event_outcomes"].items()]
        state.output.completion = f"Hidden outcomes: {hidden_outcomes}"
        agent_ids = list(game_state["agents"].keys())
        print(f"Initialized game with agents: {agent_ids}")

        # Main game loop
        for round_num in range(1, config.total_rounds + 1):
            game_state["round"] = round_num
            print(f"\n{'='*20} ROUND {round_num} {'='*20}")

            ##Summarising:
            print("Summarising context")
            await summarise_agent_states(game_state, generate)

            # Phase 1: Information and event resolution
            #conduct_information_phase(game_state)
            await conduct_information_phase(game_state, state, generate)

            # Phase 2: Communication
            print("start communication phase")
            print("Starting communication phase...")
            tasks = [
                agent_communicates(game_state, agent_id, generate)
                for agent_id in agent_ids
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            '''for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"Agent {agent_ids[i]} failed with exception: {result}")
                elif result is None:
                    print(f"Agent {agent_ids[i]} returned None")
                else:
                    print(f"Agent {agent_ids[i]} completed successfully")'''
            round_messages = [msg for msg in game_state["message_log"] if msg["round"] == round_num]
            print(f"Messages sent this round: {len(round_messages)}")
            for msg in round_messages:
                print(f"  {msg['sender']} -> {msg['recipient']}: {msg['content']}")
            #await conduct_communication_phase(game_state, state, generate)

            # Phase 3: Trading
            print("start trading phase")
            print("Starting trading phase...")
            tasks = [
                agent_trades(game_state, agent_id, generate)
                for agent_id in agent_ids
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            print("after trading")
            '''for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"Agent {agent_ids[i]} failed with exception: {result}")
                elif result is None:
                    print(f"Agent {agent_ids[i]} returned None")
                else:
                    print(f"Agent {agent_ids[i]} completed successfully")'''

            # Update state
            state.store.set("game_state", game_state)
            #display_final_results(game_state)
            print(game_state['trade_log'])



        state.output.completion += "\nGame finished."

        # Display results
        display_final_results(game_state)
        return state

    return solve

def display_final_results(game_state: GameState) -> None:
    """Display final results."""
    print(f"\n{'='*20} FINAL RESULTS {'='*20}")

    config = game_state["config"]
    insider_profits = []
    outsider_profits = []

    for agent_id, agent in game_state["agents"].items():
        profit = agent["cash"] - config.starting_cash
        agent_type = agent["type"]

        if agent_type == "insider":
            insider_profits.append(profit)
        else:
            outsider_profits.append(profit)

        print(f"  {agent_id} ({agent_type:>8}): ${agent['cash']:>8,} (profit: ${profit:>+8,})")

    # Summary statistics
    avg_insider = sum(insider_profits) / len(insider_profits) if insider_profits else 0
    avg_outsider = sum(outsider_profits) / len(outsider_profits) if outsider_profits else 0

    print(f"\nSUMMARY:")
    print(f"  Average insider profit:  ${avg_insider:>+8,.0f}")
    print(f"  Average outsider profit: ${avg_outsider:>+8,.0f}")
    print(f"  Insider advantage:       ${avg_insider - avg_outsider:>+8,.0f}")
    print(f"  Total trades:            {len(game_state['trade_log']):>8}")
    print(f"  Total messages:          {len(game_state['message_log']):>8}")

@task
def prediction_market_game(config_name: str = "baseline"):
    """Prediction market game with configurable parameters."""

    configs = {
        "baseline": GameConfig(),
        "quick_test": GameConfig(
            num_agents=4,
            total_rounds=10,
            num_events=5,
            starting_cash=5000
        ),
        "custom_probabilities": GameConfig(
            market_probabilities={
                "event_0": 0.3,  # $30 YES, $70 NO
                "event_1": 0.7,  # $70 YES, $30 NO
                "event_2": 0.2,  # $20 YES, $80 NO
            }
        )
    }

    config = configs.get(config_name, GameConfig())

    return Task(
        dataset=[Sample(
            input=f"Run prediction market game with {config_name} configuration",
            target="Game completed successfully",
            metadata={
                "config_name": config_name,
                "fixed_pricing": True
            }
        )],
        solver=run_prediction_market_game(config),
        scorer=match()
    )
