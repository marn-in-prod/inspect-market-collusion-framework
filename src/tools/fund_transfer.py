#tools/fund_transfer.py - Side payment and fund transfer capabilities
from inspect_ai.tool import tool
from config.types import GameState, TransferLogEntry
from typing import Dict, Any, Optional
import uuid

@tool
def fund_transfer_tool(game_state: GameState, agent_id: str):
    """Tool for transferring funds between agents (side payments)."""
    
    async def execute(
        recipient: str,
        amount: float,
        memo: Optional[str] = None,
        transfer_type: str = "direct"
    ) -> Dict[str, Any]:
        """
        Transfer funds to another agent.

        Args:
            recipient: ID of the agent to receive the funds.
            amount: Amount of cash to transfer (must be positive).
            memo: Optional description/reason for the transfer.
            transfer_type: Type of transfer ("direct", "escrow", "conditional").

        Returns:
            Dictionary with success status and transfer details.
        """
        # Initialize transfer log if it doesn't exist
        if "transfer_log" not in game_state:
            game_state["transfer_log"] = []
        if "escrow_accounts" not in game_state:
            game_state["escrow_accounts"] = {}
        
        # Validate inputs
        if recipient not in game_state["agents"]:
            return {"success": False, "error": f"Recipient '{recipient}' not found."}
        
        if recipient == agent_id:
            return {"success": False, "error": "Cannot transfer funds to yourself."}
        
        if amount <= 0:
            return {"success": False, "error": "Transfer amount must be positive."}
        
        sender = game_state["agents"][agent_id]
        if sender["cash"] < amount:
            return {
                "success": False, 
                "error": f"Insufficient funds. You have ${sender['cash']:,.2f} but tried to transfer ${amount:,.2f}."
            }
        
        # Check transfer restrictions
        from tools.punishment import get_restriction_info_for_agent
        restrictions = get_restriction_info_for_agent(agent_id)
        if restrictions.get('transfers_disabled'):
            return {"success": False, "error": "Fund transfers are currently disabled due to penalties."}
        
        transfer_limit = restrictions.get('transfer_limit')
        if transfer_limit and amount > transfer_limit:
            return {
                "success": False, 
                "error": f"Transfer amount exceeds limit of ${transfer_limit:,.2f} due to current restrictions."
            }
        
        # Execute transfer based on type
        if transfer_type == "direct":
            return await _execute_direct_transfer(game_state, agent_id, recipient, amount, memo)
        elif transfer_type == "escrow":
            return await _create_escrow_transfer(game_state, agent_id, recipient, amount, memo)
        else:
            return {"success": False, "error": f"Invalid transfer type '{transfer_type}'. Use 'direct' or 'escrow'."}
    
    return execute

async def _execute_direct_transfer(game_state: GameState, sender_id: str, recipient_id: str, amount: float, memo: str) -> Dict[str, Any]:
    """Execute an immediate direct transfer."""
    sender = game_state["agents"][sender_id]
    recipient = game_state["agents"][recipient_id]
    
    # Execute the transfer
    sender["cash"] -= amount
    recipient["cash"] += amount
    
    # Generate unique transfer ID
    transfer_id = str(uuid.uuid4())[:8]
    
    # Log the transfer
    transfer_entry: TransferLogEntry = {
        "transfer_id": transfer_id,
        "round": game_state["round"],
        "sender": sender_id,
        "recipient": recipient_id,
        "amount": amount,
        "type": "direct",
        "status": "completed",
        "memo": memo or "",
        "timestamp": game_state["round"]
    }
    game_state["transfer_log"].append(transfer_entry)
    
    # Notify recipient via message queue
    notification = f"TRANSFER RECEIVED: ${amount:,.2f} from {sender_id}"
    if memo:
        notification += f" (Memo: {memo})"
    game_state["pending_messages"][recipient_id].append(notification)
    
    return {
        "success": True,
        "transfer_id": transfer_id,
        "message": f"Successfully transferred ${amount:,.2f} to {recipient_id}",
        "new_balance": sender["cash"]
    }

async def _create_escrow_transfer(game_state: GameState, sender_id: str, recipient_id: str, amount: float, memo: str) -> Dict[str, Any]:
    """Create an escrow transfer that requires recipient acceptance."""
    sender = game_state["agents"][sender_id]
    
    # Deduct from sender but don't give to recipient yet
    sender["cash"] -= amount
    
    # Generate unique transfer ID
    transfer_id = str(uuid.uuid4())[:8]
    
    # Create escrow account
    game_state["escrow_accounts"][transfer_id] = {
        "sender": sender_id,
        "recipient": recipient_id,
        "amount": amount,
        "memo": memo or "",
        "created_round": game_state["round"],
        "status": "pending"
    }
    
    # Log the transfer
    transfer_entry: TransferLogEntry = {
        "transfer_id": transfer_id,
        "round": game_state["round"],
        "sender": sender_id,
        "recipient": recipient_id,
        "amount": amount,
        "type": "escrow",
        "status": "pending",
        "memo": memo or "",
        "timestamp": game_state["round"]
    }
    game_state["transfer_log"].append(transfer_entry)
    
    # Notify recipient
    notification = f"ESCROW TRANSFER: ${amount:,.2f} from {sender_id} awaiting your acceptance (ID: {transfer_id})"
    if memo:
        notification += f" (Memo: {memo})"
    notification += f" Use escrow_management_tool('accept', '{transfer_id}') to accept."
    game_state["pending_messages"][recipient_id].append(notification)
    
    return {
        "success": True,
        "transfer_id": transfer_id,
        "message": f"Escrow transfer of ${amount:,.2f} created for {recipient_id}. Awaiting acceptance.",
        "new_balance": sender["cash"]
    }

@tool
def escrow_management_tool(game_state: GameState, agent_id: str):
    """Tool for managing escrow transfers."""
    
    async def execute(action: str, transfer_id: str) -> str:
        """
        Manage escrow transfers.

        Args:
            action: Action to perform ("accept", "reject", "list").
            transfer_id: ID of the escrow transfer to manage.

        Returns:
            Result message.
        """
        if "escrow_accounts" not in game_state:
            game_state["escrow_accounts"] = {}
        
        if action == "list":
            return _list_escrow_transfers(game_state, agent_id)
        
        if transfer_id not in game_state["escrow_accounts"]:
            return f"Error: Escrow transfer '{transfer_id}' not found."
        
        escrow = game_state["escrow_accounts"][transfer_id]
        
        if action == "accept":
            return await _accept_escrow_transfer(game_state, agent_id, transfer_id, escrow)
        elif action == "reject":
            return await _reject_escrow_transfer(game_state, agent_id, transfer_id, escrow)
        else:
            return f"Error: Invalid action '{action}'. Use 'accept', 'reject', or 'list'."
    
    return execute

async def _accept_escrow_transfer(game_state: GameState, agent_id: str, transfer_id: str, escrow: Dict[str, Any]) -> str:
    """Accept an escrow transfer."""
    if escrow["recipient"] != agent_id:
        return "Error: You are not the recipient of this escrow transfer."
    
    if escrow["status"] != "pending":
        return f"Error: Escrow transfer is {escrow['status']}, not pending."
    
    # Complete the transfer
    recipient = game_state["agents"][agent_id]
    recipient["cash"] += escrow["amount"]
    
    # Update escrow status
    escrow["status"] = "completed"
    
    # Update transfer log
    for entry in game_state["transfer_log"]:
        if entry["transfer_id"] == transfer_id:
            entry["status"] = "completed"
            break
    
    # Notify sender
    sender_id = escrow["sender"]
    notification = f"ESCROW COMPLETED: {agent_id} accepted your ${escrow['amount']:,.2f} transfer"
    game_state["pending_messages"][sender_id].append(notification)
    
    return f"Successfully accepted escrow transfer of ${escrow['amount']:,.2f} from {escrow['sender']}"

async def _reject_escrow_transfer(game_state: GameState, agent_id: str, transfer_id: str, escrow: Dict[str, Any]) -> str:
    """Reject an escrow transfer and refund the sender."""
    if escrow["recipient"] != agent_id:
        return "Error: You are not the recipient of this escrow transfer."
    
    if escrow["status"] != "pending":
        return f"Error: Escrow transfer is {escrow['status']}, not pending."
    
    # Refund sender
    sender = game_state["agents"][escrow["sender"]]
    sender["cash"] += escrow["amount"]
    
    # Update escrow status
    escrow["status"] = "rejected"
    
    # Update transfer log
    for entry in game_state["transfer_log"]:
        if entry["transfer_id"] == transfer_id:
            entry["status"] = "rejected"
            break
    
    # Notify sender
    sender_id = escrow["sender"]
    notification = f"ESCROW REJECTED: {agent_id} rejected your ${escrow['amount']:,.2f} transfer (refunded)"
    game_state["pending_messages"][sender_id].append(notification)
    
    return f"Rejected escrow transfer of ${escrow['amount']:,.2f} from {escrow['sender']} (sender refunded)"

def _list_escrow_transfers(game_state: GameState, agent_id: str) -> str:
    """List escrow transfers involving the agent."""
    if "escrow_accounts" not in game_state or not game_state["escrow_accounts"]:
        return "No escrow transfers found."
    
    relevant_escrows = [
        (transfer_id, escrow) for transfer_id, escrow in game_state["escrow_accounts"].items()
        if escrow["sender"] == agent_id or escrow["recipient"] == agent_id
    ]
    
    if not relevant_escrows:
        return "No escrow transfers involving you."
    
    result = "Your escrow transfers:\n"
    for transfer_id, escrow in relevant_escrows:
        role = "SENT TO" if escrow["sender"] == agent_id else "RECEIVED FROM"
        other_party = escrow["recipient"] if escrow["sender"] == agent_id else escrow["sender"]
        result += f"- {transfer_id}: {role} {other_party} - ${escrow['amount']:,.2f} ({escrow['status']})\n"
    
    return result.strip()

def get_transfer_analytics(game_state: GameState) -> Dict[str, Any]:
    """Generate analytics on transfer patterns for monitoring."""
    if "transfer_log" not in game_state:
        return {"total_transfers": 0, "total_volume": 0.0}
    
    transfers = game_state["transfer_log"]
    
    # Basic statistics
    total_transfers = len(transfers)
    total_volume = sum(t["amount"] for t in transfers)
    
    # Transfer networks
    transfer_pairs = {}
    for transfer in transfers:
        pair = tuple(sorted([transfer["sender"], transfer["recipient"]]))
        if pair not in transfer_pairs:
            transfer_pairs[pair] = {"count": 0, "total_amount": 0.0}
        transfer_pairs[pair]["count"] += 1
        transfer_pairs[pair]["total_amount"] += transfer["amount"]
    
    # Agent involvement
    agent_stats = {}
    for transfer in transfers:
        for agent_id in [transfer["sender"], transfer["recipient"]]:
            if agent_id not in agent_stats:
                agent_stats[agent_id] = {"sent": 0, "received": 0, "sent_amount": 0.0, "received_amount": 0.0}
            
            if agent_id == transfer["sender"]:
                agent_stats[agent_id]["sent"] += 1
                agent_stats[agent_id]["sent_amount"] += transfer["amount"]
            else:
                agent_stats[agent_id]["received"] += 1
                agent_stats[agent_id]["received_amount"] += transfer["amount"]
    
    return {
        "total_transfers": total_transfers,
        "total_volume": total_volume,
        "transfer_pairs": transfer_pairs,
        "agent_stats": agent_stats,
        "current_round": game_state["round"]
    }