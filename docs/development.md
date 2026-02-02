# Development Workflow

read_when: you need setup, test, or local run instructions

## Setup

- Python 3.12+ is required.
- Create a virtual environment and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[test]
```

## Run locally

- RPC mode (JSON over stdin/stdout):

```bash
python -m pi_sdk.rpc
# or
pi-rpc
```

- Provider/model overrides:

```bash
PI_PROVIDER=openai PI_MODEL=gpt-4o-mini python -m pi_sdk.rpc
```

## Tests

- Run the test suite:

```bash
pytest
```

- If tests are missing for new behavior, add them before expanding APIs.
