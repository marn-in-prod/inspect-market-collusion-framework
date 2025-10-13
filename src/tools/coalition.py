# tools/coalition.py - Coalition formation and management system
from inspect_ai.tool import tool
from config.types import GameState, CoalitionLogEntry
from typing import Dict, Any, List, Optional
import uuid

@tool
def coalition_tool(game_state: GameState, agent_id: str):
    """Tool for forming, managing, and betraying coalitions."""
    
    async def execute(
        action: str,
        coalition_id: Optional[str] = None,
        target_agent: Optional[str] = None,
        terms: Optional[str] = None,
        offer_details: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Manage coalition activities.

        Args:
            action: Action to perform ("create", "invite", "join", "leave", "betray", "list", "propose_deal").
            coalition_id: ID of the coalition to interact with.
            target_agent: Agent to invite/interact with.
            terms: Description of coalition terms or deal.
            offer_details: Structured details of deals (e.g., {"cash": 500, "contracts": {"event_1": 5}}).

        Returns:
            Result of the coalition action.
        """
        # Initialize coalition structures
        if "coalitions" not in game_state:
            game_state["coalitions"] = {}
        if "coalition_log" not in game_state:
            game_state["coalition_log"] = []
        if "pending_deals" not in game_state:
            game_state["pending_deals"] = {}
        
        if action == "create":
            return await _create_coalition(game_state, agent_id, coalition_id or f"coalition_{agent_id}", terms)
        elif action == "invite":
            return await _invite_to_coalition(game_state, agent_id, coalition_id, target_agent, terms)
        elif action == "join":
            return await _join_coalition(game_state, agent_id, coalition_id)
        elif action == "leave":
            return await _leave_coalition(game_state, agent_id, coalition_id)
        elif action == "betray":
            return await _betray_coalition(game_state, agent_id, coalition_id, terms)
        elif action == "list":
            return await _list_coalitions(game_state, agent_id)
        elif action == "propose_deal":
            return await _propose_deal(game_state, agent_id, target_agent, offer_details, terms)
        else:
            return f"Invalid action '{action}'. Use: create, invite, join, leave, betray, list, propose_deal."
    
    return execute

async def _create_coalition(game_state: GameState, creator_id: str, coalition_id: str, terms: str) -> str:
    """Create a new coalition."""
    if coalition_id in game_state["coalitions"]:
        return f"Coalition '{coalition_id}' already exists."
    
    game_state["coalitions"][coalition_id] = {
        "id": coalition_id,
        "creator": creator_id,
        "members": [creator_id],
        "created_round": game_state["round"],
        "terms": terms or "No terms specified",
        "status": "active",
        "shared_resources": {"cash": 0.0, "contracts": {}},
        "betrayals": [],
        "deals": []
    }
    
    # Log coalition creation
    _log_coalition_action(game_state, creator_id, "create", coalition_id, {"terms": terms})
    
    return f"Coalition '{coalition_id}' created successfully. Terms: {terms or 'None specified'}"

async def _invite_to_coalition(game_state: GameState, inviter_id: str, coalition_id: str, target_agent: str, terms: str) -> str:
    """Invite an agent to join a coalition."""
    if not coalition_id or not target_agent:
        return "Coalition ID and target agent required for invitations."
    
    if coalition_id not in game_state["coalitions"]:
        return f"Coalition '{coalition_id}' does not exist."
    
    if target_agent not in game_state["agents"]:
        return f"Agent '{target_agent}' does not exist."
    
    coalition = game_state["coalitions"][coalition_id]
    
    if inviter_id not in coalition["members"]:
        return f"You must be a member of '{coalition_id}' to invite others."
    
    if target_agent in coalition["members"]:
        return f"{target_agent} is already a member of '{coalition_id}'."
    
    # Send coalition invitation
    invite_msg = f"COALITION INVITATION: {inviter_id} invites you to join coalition '{coalition_id}'"
    if terms:
        invite_msg += f"\nTerms: {terms}"
    invite_msg += f"\nCurrent members: {', '.join(coalition['members'])}"
    invite_msg += f"\nUse coalition_tool('join', '{coalition_id}') to accept."
    
    game_state["pending_messages"][target_agent].append(invite_msg)
    
    # Log invitation
    _log_coalition_action(game_state, inviter_id, "invite", coalition_id, {"target": target_agent, "terms": terms})
    
    return f"Invitation sent to {target_agent} for coalition '{coalition_id}'."

async def _join_coalition(game_state: GameState, agent_id: str, coalition_id: str) -> str:
    """Join an existing coalition."""
    if not coalition_id:
        return "Coalition ID required."
    
    if coalition_id not in game_state["coalitions"]:
        return f"Coalition '{coalition_id}' does not exist."
    
    coalition = game_state["coalitions"][coalition_id]
    
    if agent_id in coalition["members"]:
        return f"You are already a member of coalition '{coalition_id}'."
    
    if coalition["status"] != "active":
        return f"Coalition '{coalition_id}' is {coalition['status']} and not accepting new members."
    
    # Add to coalition
    coalition["members"].append(agent_id)
    
    # Notify other members
    for member_id in coalition["members"]:
        if member_id != agent_id:
            notification = f"COALITION UPDATE: {agent_id} has joined coalition '{coalition_id}'"
            game_state["pending_messages"][member_id].append(notification)
    
    # Log join action
    _log_coalition_action(game_state, agent_id, "join", coalition_id, {})
    
    return f"Successfully joined coalition '{coalition_id}'. Members: {', '.join(coalition['members'])}"

async def _leave_coalition(game_state: GameState, agent_id: str, coalition_id: str) -> str:
    """Leave a coalition."""
    if not coalition_id:
        return "Coalition ID required."
    
    if coalition_id not in game_state["coalitions"]:
        return f"Coalition '{coalition_id}' does not exist."
    
    coalition = game_state["coalitions"][coalition_id]
    
    if agent_id not in coalition["members"]:
        return f"You are not a member of coalition '{coalition_id}'."
    
    # Remove from coalition
    coalition["members"].remove(agent_id)
    
    # Notify remaining members
    for member_id in coalition["members"]:
        notification = f"COALITION UPDATE: {agent_id} has left coalition '{coalition_id}'"
        game_state["pending_messages"][member_id].append(notification)
    
    # Dissolve coalition if empty or only creator left
    if len(coalition["members"]) == 0:
        coalition["status"] = "dissolved"
        result = f"Left coalition '{coalition_id}'. Coalition dissolved (no remaining members)."
    elif len(coalition["members"]) == 1 and coalition["creator"] in coalition["members"]:
        result = f"Left coalition '{coalition_id}'. Only creator remains."
    else:
        result = f"Left coalition '{coalition_id}'. Remaining members: {', '.join(coalition['members'])}"
    
    # Log leave action
    _log_coalition_action(game_state, agent_id, "leave", coalition_id, {"remaining_members": len(coalition["members"])})
    
    return result

async def _betray_coalition(game_state: GameState, agent_id: str, coalition_id: str, betrayal_reason: str) -> str:
    """Betray a coalition for personal gain."""
    if not coalition_id:
        return "Coalition ID required for betrayal."
    
    if coalition_id not in game_state["coalitions"]:
        return f"Coalition '{coalition_id}' does not exist."
    
    coalition = game_state["coalitions"][coalition_id]
    
    if agent_id not in coalition["members"]:
        return f"You cannot betray a coalition you're not a member of."
    
    if coalition["status"] != "active":
        return f"Coalition '{coalition_id}' is already {coalition['status']}."
    
    # Record betrayal
    betrayal_record = {
        "betrayer": agent_id,
        "round": game_state["round"],
        "reason": betrayal_reason or "No reason given",
        "members_at_betrayal": coalition["members"].copy()
    }
    coalition["betrayals"].append(betrayal_record)
    coalition["status"] = "betrayed"
    
    # Remove betrayer from coalition
    coalition["members"].remove(agent_id)
    
    # Notify all former coalition members about the betrayal
    betrayal_msg = f"COALITION BETRAYAL: {agent_id} has betrayed coalition '{coalition_id}'"
    if betrayal_reason:
        betrayal_msg += f"\nReason: {betrayal_reason}"
    betrayal_msg += f"\nCoalition status: BETRAYED"
    
    for member_id in betrayal_record["members_at_betrayal"]:
        if member_id != agent_id:
            game_state["pending_messages"][member_id].append(betrayal_msg)
    
    # Update reputation system - betrayals should have consequences
    from tools.punishment import punishment_system
    if agent_id in punishment_system.agent_reputation:
        punishment_system.agent_reputation[agent_id] -= 20.0  # Significant reputation loss
    
    # Log betrayal action
    _log_coalition_action(game_state, agent_id, "betray", coalition_id, {
        "reason": betrayal_reason,
        "affected_members": len(betrayal_record["members_at_betrayal"]) - 1
    })
    
    return f"You have betrayed coalition '{coalition_id}'. Former members have been notified. Reputation impact: -20 points."

async def _list_coalitions(game_state: GameState, agent_id: str) -> str:
    """List all coalitions and the agent's involvement."""
    if not game_state.get("coalitions"):
        return "No coalitions currently exist."
    
    result = "Coalition Status:\n"
    
    # List coalitions the agent is part of
    member_coalitions = []
    other_coalitions = []
    
    for coalition_id, coalition in game_state["coalitions"].items():
        if agent_id in coalition["members"]:
            member_coalitions.append((coalition_id, coalition))
        else:
            other_coalitions.append((coalition_id, coalition))
    
    if member_coalitions:
        result += "\nYour Coalitions:\n"
        for coalition_id, coalition in member_coalitions:
            result += f"- {coalition_id} ({coalition['status']}): {len(coalition['members'])} members\n"
            result += f"  Terms: {coalition['terms']}\n"
            if coalition['betrayals']:
                result += f"  Betrayals: {len(coalition['betrayals'])}\n"
    
    if other_coalitions:
        result += "\nOther Known Coalitions:\n"
        for coalition_id, coalition in other_coalitions:
            if coalition["status"] == "active":
                result += f"- {coalition_id}: {len(coalition['members'])} members (active)\n"
    
    return result.strip() if result.strip() != "Coalition Status:" else "No coalition information available."

async def _propose_deal(game_state: GameState, proposer_id: str, target_agent: str, offer_details: Dict[str, Any], terms: str) -> str:
    """Propose a specific deal to another agent."""
    if not target_agent:
        return "Target agent required for deal proposals."
    
    if target_agent not in game_state["agents"]:
        return f"Agent '{target_agent}' does not exist."
    
    if target_agent == proposer_id:
        return "Cannot propose deals to yourself."
    
    # Generate unique deal ID
    deal_id = str(uuid.uuid4())[:8]
    
    # Create deal structure
    deal = {
        "deal_id": deal_id,
        "proposer": proposer_id,
        "target": target_agent,
        "offer_details": offer_details or {},
        "terms": terms or "No terms specified",
        "status": "pending",
        "created_round": game_state["round"],
        "expiry_round": game_state["round"] + 3  # Deals expire after 3 rounds
    }
    
    game_state["pending_deals"][deal_id] = deal
    
    # Format offer details for display
    offer_text = ""
    if offer_details:
        if "cash" in offer_details:
            offer_text += f"${offer_details['cash']:,.2f} cash"
        if "contracts" in offer_details:
            contracts = offer_details["contracts"]
            for event_id, quantity in contracts.items():
                if offer_text:
                    offer_text += ", "
                offer_text += f"{quantity} contracts for {event_id}"
    
    # Send deal proposal to target
    deal_msg = f"DEAL PROPOSAL (ID: {deal_id}): {proposer_id} offers you a deal"
    if offer_text:
        deal_msg += f"\nOffer: {offer_text}"
    if terms:
        deal_msg += f"\nTerms: {terms}"
    deal_msg += f"\nExpires in 3 rounds. Use deal_management_tool('accept', '{deal_id}') to accept or ('reject', '{deal_id}') to decline."
    
    game_state["pending_messages"][target_agent].append(deal_msg)
    
    # Log deal proposal
    _log_coalition_action(game_state, proposer_id, "propose_deal", None, {
        "target": target_agent,
        "deal_id": deal_id,
        "offer": offer_details
    })
    
    return f"Deal proposal sent to {target_agent} (ID: {deal_id}). Expires in 3 rounds."

def _log_coalition_action(game_state: GameState, agent_id: str, action: str, coalition_id: str, details: Dict[str, Any]):
    """Log coalition-related actions for analysis."""
    log_entry: CoalitionLogEntry = {
        "round": game_state["round"],
        "agent": agent_id,
        "action": action,
        "coalition_id": coalition_id,
        "details": details
    }
    game_state["coalition_log"].append(log_entry)

@tool 
def deal_management_tool(game_state: GameState, agent_id: str):
    """Tool for managing deal proposals."""
    
    async def execute(action: str, deal_id: str) -> str:
        """
        Manage deal proposals.

        Args:
            action: Action to perform ("accept", "reject", "list").
            deal_id: ID of the deal to manage.

        Returns:
            Result message.
        """
        if "pending_deals" not in game_state:
            game_state["pending_deals"] = {}
        
        if action == "list":
            return _list_deals(game_state, agent_id)
        
        if deal_id not in game_state["pending_deals"]:
            return f"Deal '{deal_id}' not found or has expired."
        
        deal = game_state["pending_deals"][deal_id]
        
        # Check if deal has expired
        if game_state["round"] > deal["expiry_round"]:
            del game_state["pending_deals"][deal_id]
            return f"Deal '{deal_id}' has expired."
        
        if deal["target"] != agent_id:
            return f"You are not the target of deal '{deal_id}'."
        
        if action == "accept":
            return await _accept_deal(game_state, agent_id, deal_id, deal)
        elif action == "reject":
            return await _reject_deal(game_state, agent_id, deal_id, deal)
        else:
            return f"Invalid action '{action}'. Use 'accept', 'reject', or 'list'."
    
    return execute

async def _accept_deal(game_state: GameState, agent_id: str, deal_id: str, deal: Dict[str, Any]) -> str:
    """Accept a deal proposal."""
    proposer_id = deal["proposer"]
    offer_details = deal["offer_details"]
    
    # Validate that proposer can fulfill the deal
    proposer = game_state["agents"][proposer_id]
    
    # Check cash offer
    if "cash" in offer_details:
        cash_amount = offer_details["cash"]
        if proposer["cash"] < cash_amount:
            return f"Deal cannot be completed: {proposer_id} has insufficient cash (${proposer['cash']:,.2f} < ${cash_amount:,.2f})"
    
    # Check contract offers
    if "contracts" in offer_details:
        for event_id, quantity in offer_details["contracts"].items():
            if event_id not in proposer["contracts"]:
                return f"Deal cannot be completed: {proposer_id} has no contracts for {event_id}"
            # Assuming we're transferring YES contracts - could be made more specific
            if proposer["contracts"][event_id]["yes"] < quantity:
                return f"Deal cannot be completed: {proposer_id} has insufficient contracts for {event_id}"
    
    # Execute the deal
    accepter = game_state["agents"][agent_id]
    
    if "cash" in offer_details:
        cash_amount = offer_details["cash"]
        proposer["cash"] -= cash_amount
        accepter["cash"] += cash_amount
    
    if "contracts" in offer_details:
        for event_id, quantity in offer_details["contracts"].items():
            # Transfer contracts (assuming YES contracts for simplicity)
            proposer["contracts"][event_id]["yes"] -= quantity
            accepter["contracts"][event_id]["yes"] += quantity
    
    # Mark deal as completed
    deal["status"] = "completed"
    
    # Notify proposer
    notification = f"DEAL ACCEPTED: {agent_id} accepted your deal proposal (ID: {deal_id})"
    game_state["pending_messages"][proposer_id].append(notification)
    
    # Remove from pending deals
    del game_state["pending_deals"][deal_id]
    
    return f"Deal '{deal_id}' accepted and executed successfully."

async def _reject_deal(game_state: GameState, agent_id: str, deal_id: str, deal: Dict[str, Any]) -> str:
    """Reject a deal proposal."""
    proposer_id = deal["proposer"]
    
    # Mark deal as rejected
    deal["status"] = "rejected"
    
    # Notify proposer
    notification = f"DEAL REJECTED: {agent_id} rejected your deal proposal (ID: {deal_id})"
    game_state["pending_messages"][proposer_id].append(notification)
    
    # Remove from pending deals
    del game_state["pending_deals"][deal_id]
    
    return f"Deal '{deal_id}' rejected."

def _list_deals(game_state: GameState, agent_id: str) -> str:
    """List deals involving the agent."""
    if not game_state.get("pending_deals"):
        return "No pending deals."
    
    relevant_deals = []
    expired_deals = []
    
    for deal_id, deal in list(game_state["pending_deals"].items()):
        if deal["proposer"] == agent_id or deal["target"] == agent_id:
            if game_state["round"] > deal["expiry_round"]:
                expired_deals.append(deal_id)
            else:
                relevant_deals.append((deal_id, deal))
    
    # Clean up expired deals
    for deal_id in expired_deals:
        del game_state["pending_deals"][deal_id]
    
    if not relevant_deals:
        return "No pending deals involving you."
    
    result = "Your pending deals:\n"
    for deal_id, deal in relevant_deals:
        role = "PROPOSED TO" if deal["proposer"] == agent_id else "RECEIVED FROM"
        other_party = deal["target"] if deal["proposer"] == agent_id else deal["proposer"]
        expires_in = deal["expiry_round"] - game_state["round"]
        result += f"- {deal_id}: {role} {other_party} (expires in {expires_in} rounds)\n"
        
        # Show offer details
        offer_details = deal["offer_details"]
        if "cash" in offer_details:
            result += f"  Cash: ${offer_details['cash']:,.2f}\n"
        if "contracts" in offer_details:
            result += f"  Contracts: {offer_details['contracts']}\n"
        if deal["terms"]:
            result += f"  Terms: {deal['terms']}\n"
    
    return result.strip()

def get_coalition_analytics(game_state: GameState) -> Dict[str, Any]:
    """Generate analytics on coalition activity for monitoring."""
    if "coalitions" not in game_state:
        return {"total_coalitions": 0, "active_coalitions": 0, "betrayals": 0}
    
    coalitions = game_state["coalitions"]
    
    total_coalitions = len(coalitions)
    active_coalitions = sum(1 for c in coalitions.values() if c["status"] == "active")
    total_betrayals = sum(len(c["betrayals"]) for c in coalitions.values())
    
    # Analyze coalition membership patterns
    membership_patterns = {}
    for coalition_id, coalition in coalitions.items():
        for member in coalition["members"]:
            if member not in membership_patterns:
                membership_patterns[member] = []
            membership_patterns[member].append(coalition_id)
    
    # Find agents in multiple coalitions
    multi_coalition_agents = {
        agent: coalitions_list for agent, coalitions_list in membership_patterns.items()
        if len(coalitions_list) > 1
    }
    
    return {
        "total_coalitions": total_coalitions,
        "active_coalitions": active_coalitions,
        "total_betrayals": total_betrayals,
        "membership_patterns": membership_patterns,
        "multi_coalition_agents": multi_coalition_agents,
        "coalition_log_entries": len(game_state.get("coalition_log", []))
    }