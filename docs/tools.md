# Tools

read_when: you need tool behavior details or are adding custom tool implementations

## Default tools

- `read`
  - Inputs: `path`, optional `offset` (1-indexed), optional `limit`.
  - Text output is truncated to 2000 lines or 50KB (whichever hits first).
  - Images (png/jpg/gif/webp) are returned as attachments; auto-resizes to 2000x2000 if Pillow is available.
  - Truncation metadata is returned in `details.truncation`.

- `write`
  - Inputs: `path`, `content`.
  - Creates parent directories automatically and overwrites existing files.
  - Returns a success message with byte count.

- `edit`
  - Inputs: `path`, `old_text`, `new_text`.
  - Performs exact replacement; falls back to fuzzy matching when needed.
  - Preserves BOM/CRLF handling and returns a unified diff summary.

- `bash`
  - Inputs: `command`, optional `timeout` (seconds).
  - Streams stdout/stderr; output is truncated to the last 2000 lines or 50KB.
  - If truncated, full output is written to a temp file and reported in details.
  - Supports command prefixes and spawn hooks for custom shells.

## Custom operations

Each tool accepts `operations` for delegating file access or shell execution to
remote systems (SSH, containers, sandboxes). The interface mirrors the default
filesystem/shell behavior to keep results compatible with the agent loop.
