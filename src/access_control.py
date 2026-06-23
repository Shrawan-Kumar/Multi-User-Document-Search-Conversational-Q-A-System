"""
access_control.py
------------------
Simple, explicit access-control layer. Deliberately kept separate from
retrieval logic so it's easy to point to in an interview: "here is the
ONE function that decides what a user is allowed to see."
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


class AccessDeniedError(Exception):
    """Raised when a user has no registered access at all."""
    pass


def get_allowed_companies(user_email: str) -> list[str]:
    """
    Return the list of companies a given user is permitted to query.
    Raises AccessDeniedError if the email isn't registered at all.
    """
    user_email = user_email.strip().lower()
    if user_email not in config.USER_ACCESS_MAP:
        raise AccessDeniedError(
            f"'{user_email}' is not a recognized user. "
            f"Registered demo users: {list(config.USER_ACCESS_MAP.keys())}"
        )
    return config.USER_ACCESS_MAP[user_email]


def build_faiss_filter(user_email: str):
    """
    Build a metadata filter dict compatible with LangChain's FAISS
    similarity_search(filter=...) API. FAISS's filter does an exact-match
    (or `in`, for lists) check against document metadata, so we restrict
    retrieval to only the companies this user is allowed to see *before*
    any chunk content reaches the LLM. This is what guarantees isolation
    — unauthorized chunks are never retrieved, never enter the prompt,
    and never reach the user.
    """
    allowed_companies = get_allowed_companies(user_email)

    def _filter(metadata: dict) -> bool:
        return metadata.get("company") in allowed_companies

    return _filter, allowed_companies


def list_registered_users() -> list[str]:
    return list(config.USER_ACCESS_MAP.keys())