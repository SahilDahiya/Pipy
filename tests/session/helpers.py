from __future__ import annotations


def user_msg(text: str):
    return {"role": "user", "content": text, "timestamp": 1}


def assistant_msg(text: str):
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "api": "anthropic-messages",
        "provider": "anthropic",
        "model": "claude-test",
        "usage": {
            "input": 1,
            "output": 1,
            "cache_read": 0,
            "cache_write": 0,
            "total_tokens": 2,
            "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
        },
        "stop_reason": "stop",
        "timestamp": 1,
    }
