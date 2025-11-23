from typing import Dict, List, Tuple
from collections import Counter


def count_votes(votes: Dict[str, str]) -> Tuple[str, int, bool]:
    """
    Count votes and determine winner

    Args:
        votes: Dictionary mapping agent_id to viewpoint

    Returns:
        Tuple of (winning_viewpoint, vote_count, is_tie)
    """
    if not votes:
        return "", 0, False

    vote_counts = Counter(votes.values())
    max_count = max(vote_counts.values())
    winners = [v for v, c in vote_counts.items() if c == max_count]

    is_tie = len(winners) > 1
    winning_viewpoint = winners[0] if not is_tie else ""

    return winning_viewpoint, max_count, is_tie


def get_tied_viewpoints(votes: Dict[str, str]) -> List[str]:
    """
    Get list of viewpoints that are tied

    Args:
        votes: Dictionary mapping agent_id to viewpoint

    Returns:
        List of viewpoints with maximum votes
    """
    if not votes:
        return []

    vote_counts = Counter(votes.values())
    max_count = max(vote_counts.values())
    return [v for v, c in vote_counts.items() if c == max_count]


def get_vote_distribution(votes: Dict[str, str]) -> Dict[str, int]:
    """
    Get distribution of votes

    Args:
        votes: Dictionary mapping agent_id to viewpoint

    Returns:
        Dictionary mapping viewpoint to vote count
    """
    return dict(Counter(votes.values()))
