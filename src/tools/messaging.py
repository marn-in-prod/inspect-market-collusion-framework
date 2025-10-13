# tools/messaging.py - Enhanced with group chat, public forum, and anonymity
from inspect_ai.tool import tool
from config.types import GameState
from typing import List, Optional
from tools.punishment import get_restriction_info_for_agent
import random

@tool
def messaging_tool(game_state: GameState, agent_id: str):
    """Enhanced messaging tool supporting private, group, and public communications with anonymity."""
    
    async def execute(
        message: str, 
        recipient: Optional[str] = None,
        channel: str = "private",
        group_id: Optional[str] = None,
        anonymous: bool = False
    ) -> str:
        """
        Send a message through various communication channels.

        Args:
            message: The content of the message you are sending.
            recipient: For private messages, the ID of the recipient agent.
            channel: Communication channel ("private", "group", "public").
            group_id: For group messages, the ID of the group to send to.
            anonymous: Whether to send the message anonymously.

        Returns:
            A confirmation or error message.
        """
        config = game_state["config"]
        
        # Check message limits per round
        messages_sent_this_round = sum(
            1 for msg in game_state["message_log"]
            if msg["sender"] == agent_id and msg["round"] == game_state["round"]
        )
        message_limit = config.communication.message_limits.get("per_round", 3)
        if messages_sent_this_round >= message_limit:
            return f"Error: Message limit of {message_limit} per round exceeded."
        
        # Check character limit and current restrictions        
        restrictions = get_restriction_info_for_agent(agent_id)

        char_limit = restrictions.get('message_char_limit') or config.communication.char_limit
        if char_limit and len(message) > char_limit:
            return f"Error: Message exceeds character limit of {char_limit}."
        
        # Initialize communication structures if they don't exist
        if "groups" not in game_state:
            game_state["groups"] = {}
        if "public_messages" not in game_state:
            game_state["public_messages"] = []
        
        # Handle different channel types
        if channel == "private":
            return await _send_private_message(game_state, agent_id, recipient, message, anonymous)
        elif channel == "group":
            return await _send_group_message(game_state, agent_id, group_id, message, anonymous)
        elif channel == "public":
            return await _send_public_message(game_state, agent_id, message, anonymous)
        else:
            return f"Error: Invalid channel '{channel}'. Use 'private', 'group', or 'public'."
    
    return execute

async def _send_private_message(game_state: GameState, sender_id: str, recipient: str, message: str, anonymous: bool) -> str:
    """Handle private message sending."""
    if not recipient:
        return "Error: Recipient required for private messages."
    
    if recipient not in game_state["agents"]:
        return f"Error: Invalid recipient '{recipient}'."
    
    if recipient == sender_id:
        return "Error: You cannot send a message to yourself."
    
    # Format sender name
    sender_name = "Anonymous" if anonymous else sender_id
    formatted_message = f"From {sender_name}: {message}"
    
    # Queue for delivery
    game_state["pending_messages"][recipient].append(formatted_message)
    
    # Log the message
    _log_message(game_state, sender_id, "private", message, recipients=[recipient], anonymous=anonymous)
    
    return f"Private message sent to {recipient}" + (" (anonymously)" if anonymous else "")

async def _send_group_message(game_state: GameState, sender_id: str, group_id: str, message: str, anonymous: bool) -> str:
    """Handle group message sending."""
    if not group_id:
        return "Error: Group ID required for group messages."
    
    if group_id not in game_state["groups"]:
        return f"Error: Group '{group_id}' does not exist. Use create_group() first."
    
    group = game_state["groups"][group_id]
    if sender_id not in group["members"]:
        return f"Error: You are not a member of group '{group_id}'."
    
    # Format sender name
    sender_name = "Anonymous Member" if anonymous else sender_id
    formatted_message = f"[{group_id}] {sender_name}: {message}"
    
    # Send to all group members except sender
    recipients = [member for member in group["members"] if member != sender_id]
    for recipient in recipients:
        game_state["pending_messages"][recipient].append(formatted_message)
    
    # Log the message
    _log_message(game_state, sender_id, "group", message, group_id=group_id, recipients=recipients, anonymous=anonymous)
    
    return f"Message sent to group '{group_id}' ({len(recipients)} recipients)" + (" (anonymously)" if anonymous else "")

async def _send_public_message(game_state: GameState, sender_id: str, message: str, anonymous: bool) -> str:
    """Handle public forum message sending."""
    # Format sender name
    sender_name = "Anonymous" if anonymous else sender_id
    formatted_message = f"[PUBLIC] {sender_name}: {message}"
    
    # Add to public forum
    public_msg = {
        "round": game_state["round"],
        "sender": sender_id if not anonymous else "anonymous",
        "message": message,
        "anonymous": anonymous
    }
    game_state["public_messages"].append(public_msg)
    
    # Deliver to all other agents
    recipients = [agent_id for agent_id in game_state["agents"] if agent_id != sender_id]
    for recipient in recipients:
        game_state["pending_messages"][recipient].append(formatted_message)
    
    # Log the message
    _log_message(game_state, sender_id, "public", message, recipients=recipients, anonymous=anonymous)
    
    return f"Message posted to public forum" + (" (anonymously)" if anonymous else "")

def _log_message(game_state: GameState, sender_id: str, channel: str, message: str, 
                group_id: str = None, recipients: List[str] = None, anonymous: bool = False):
    """Log message for monitoring and analysis."""
    log_entry = {
        "round": game_state["round"],
        "sender": sender_id,
        "channel": channel,
        "content": message,
        "anonymous": anonymous,
        "recipients": recipients or [],
        "group_id": group_id
    }
    game_state["message_log"].append(log_entry)

@tool
def group_management_tool(game_state: GameState, agent_id: str):
    """Tool for creating and managing communication groups."""
    
    async def execute(
        action: str,
        group_id: str,
        target_agent: Optional[str] = None
    ) -> str:
        """
        Manage communication groups.

        Args:
            action: Action to perform ("create", "join", "leave", "invite", "list").
            group_id: ID of the group to manage.
            target_agent: For invite actions, the agent to invite.

        Returns:
            Confirmation or error message.
        """
        # Initialize groups if needed
        if "groups" not in game_state:
            game_state["groups"] = {}
        
        if action == "create":
            return await _create_group(game_state, agent_id, group_id)
        elif action == "join":
            return await _join_group(game_state, agent_id, group_id)
        elif action == "leave":
            return await _leave_group(game_state, agent_id, group_id)
        elif action == "invite":
            return await _invite_to_group(game_state, agent_id, group_id, target_agent)
        elif action == "list":
            return await _list_groups(game_state, agent_id)
        else:
            return f"Error: Invalid action '{action}'. Use: create, join, leave, invite, list."
    
    return execute

async def _create_group(game_state: GameState, creator_id: str, group_id: str) -> str:
    """Create a new communication group."""
    if group_id in game_state["groups"]:
        return f"Error: Group '{group_id}' already exists."
    
    game_state["groups"][group_id] = {
        "creator": creator_id,
        "members": [creator_id],
        "created_round": game_state["round"]
    }
    
    return f"Group '{group_id}' created successfully. You are the first member."

async def _join_group(game_state: GameState, agent_id: str, group_id: str) -> str:
    """Join an existing group."""
    if group_id not in game_state["groups"]:
        return f"Error: Group '{group_id}' does not exist."
    
    group = game_state["groups"][group_id]
    if agent_id in group["members"]:
        return f"You are already a member of group '{group_id}'."
    
    group["members"].append(agent_id)
    return f"Successfully joined group '{group_id}'. Members: {len(group['members'])}"

async def _leave_group(game_state: GameState, agent_id: str, group_id: str) -> str:
    """Leave a group."""
    if group_id not in game_state["groups"]:
        return f"Error: Group '{group_id}' does not exist."
    
    group = game_state["groups"][group_id]
    if agent_id not in group["members"]:
        return f"You are not a member of group '{group_id}'."
    
    group["members"].remove(agent_id)
    
    # Delete group if empty
    if not group["members"]:
        del game_state["groups"][group_id]
        return f"Left group '{group_id}'. Group was deleted (no remaining members)."
    
    return f"Left group '{group_id}'. Remaining members: {len(group['members'])}"

async def _invite_to_group(game_state: GameState, inviter_id: str, group_id: str, target_agent: str) -> str:
    """Invite another agent to join a group."""
    if not target_agent:
        return "Error: Target agent required for invitations."
    
    if target_agent not in game_state["agents"]:
        return f"Error: Agent '{target_agent}' does not exist."
    
    if group_id not in game_state["groups"]:
        return f"Error: Group '{group_id}' does not exist."
    
    group = game_state["groups"][group_id]
    if inviter_id not in group["members"]:
        return f"Error: You must be a member of '{group_id}' to invite others."
    
    if target_agent in group["members"]:
        return f"{target_agent} is already a member of '{group_id}'."
    
    # Send invitation as a private message
    invite_msg = f"INVITATION: {inviter_id} has invited you to join group '{group_id}'. Use group_management_tool('join', '{group_id}') to accept."
    game_state["pending_messages"][target_agent].append(invite_msg)
    
    return f"Invitation sent to {target_agent} for group '{group_id}'."

async def _list_groups(game_state: GameState, agent_id: str) -> str:
    """List available groups and membership status."""
    if "groups" not in game_state or not game_state["groups"]:
        return "No groups currently exist."
    
    result = "Available groups:\n"
    for group_id, group_info in game_state["groups"].items():
        is_member = agent_id in group_info["members"]
        member_count = len(group_info["members"])
        result += f"- {group_id}: {member_count} members"
        if is_member:
            result += " (YOU ARE A MEMBER)"
        result += "\n"
    
    return result.strip()