"""Session tree utilities for branching conversations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SessionNode:
    node_id: str
    parent_id: Optional[str]
    children: List[str] = field(default_factory=list)


class SessionTree:
    def __init__(self) -> None:
        self._nodes: Dict[str, SessionNode] = {}
        self._root_id: Optional[str] = None

    def add_root(self, node_id: str) -> None:
        if self._root_id is not None:
            raise ValueError("Root already exists")
        self._root_id = node_id
        self._nodes[node_id] = SessionNode(node_id=node_id, parent_id=None)

    def add_child(self, parent_id: str, node_id: str) -> None:
        if parent_id not in self._nodes:
            raise KeyError(f"Parent not found: {parent_id}")
        if node_id in self._nodes:
            raise ValueError(f"Node already exists: {node_id}")
        self._nodes[node_id] = SessionNode(node_id=node_id, parent_id=parent_id)
        self._nodes[parent_id].children.append(node_id)

    def get_node(self, node_id: str) -> SessionNode:
        if node_id not in self._nodes:
            raise KeyError(f"Node not found: {node_id}")
        return self._nodes[node_id]

    def root_id(self) -> Optional[str]:
        return self._root_id

    def to_dict(self) -> Dict[str, dict]:
        return {
            node_id: {
                "parent_id": node.parent_id,
                "children": list(node.children),
            }
            for node_id, node in self._nodes.items()
        }
