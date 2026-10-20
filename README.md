# Restaurant Order Agent

A LangGraph agent that takes a food order, checks stock, cooks and serves it
(with simulated failures and retries), prints a bill and takes payment.
Uses Groq for order extraction.

## Features

- Order several dishes in one message ("2 pizzas and 1 burger") or from the tray in the web UI
- Ask for the menu ("show me the menu") to see prices and stock
- Partial stock: accept the adjusted order or reject it (3 order attempts)
- Cook and serve with simulated failures and retries
- Bill with subtotal and 5% GST
- Demo payment by UPI, card or cash (no real payment details are collected)

Flow: `show_menu → get_order → extract_order → order_confirm → (partial_decision) → cook → serve → generate_bill → payment`

## Setup

Put `GROQ_API_KEY=...` in `.env`, then install dependencies into the venv:

```bash
uv pip install --python venv/bin/python -r pyproject.toml
```

## Web UI

```bash
venv/bin/uvicorn server:api --reload
```

Open http://127.0.0.1:8000

## CLI

```bash
venv/bin/python main.py
```

Set `COOK_TEST_RESULTS` / `SERVE_TEST_RESULTS` in `main.py` to lists of
booleans for deterministic cook/serve outcomes.
