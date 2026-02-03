# Sessions

read_when: you are persisting runs, branching, or inspecting JSONL history

## JSONL format

- Each session file begins with a `session` header containing `id`, `cwd`, `timestamp`, and `version`.
- Entries are append-only and include `id`, `parentId`, and `timestamp` for tree navigation.
- Python dataclasses expose these fields as `parent_id`, `thinking_level`, and `model_id`, while the JSONL stays camelCase for compatibility.
- Supported entry types: `message`, `thinking_level_change`, `model_change`, `compaction`, `branch_summary`,
  `custom`, `custom_message`, `label`, `session_info`.

## Default location

- Session files live under `~/.pi/agent/sessions/--<cwd>--/`.
- Override the root with `PI_CODING_AGENT_DIR`.

## SessionManager highlights

- `create()`, `open(path)`, `continue_recent()` for lifecycle management.
- `append_message()` and related append helpers for other entry types.
- `branch()` and `branch_with_summary()` for creating new paths.
- `get_tree()` and `get_branch()` for traversal.
- `create_branched_session(leaf_id)` to extract a single path.
- `build_session_context()` to reconstruct LLM-ready context for a leaf.
- `list()` and `list_all()` for session discovery.
