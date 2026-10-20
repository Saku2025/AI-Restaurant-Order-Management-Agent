import os
import random
import uuid
from typing import Annotated, TypedDict, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=GROQ_API_KEY,
)

# Price in rupees and stock level. Stock is decremented when an order is served.
MENU = {
    "pizza": {"price": 249, "stock": 5},
    "burger": {"price": 149, "stock": 10},
    "pasta": {"price": 199, "stock": 3},
    "biryani": {"price": 279, "stock": 0},
}

GST_RATE = 0.05

# Unrelated / unparseable messages can loop get_order -> extract_order many
# times; LangGraph's default recursion limit (25) would crash the run.
RECURSION_LIMIT = 200


class OrderLine(BaseModel):
    dish_name: str = Field(description="Dish name, singular and lowercase.")
    quantity: int = Field(description="Requested quantity of this dish.")


class OrderExtraction(BaseModel):
    is_food_order: bool = Field(
        description="True if the user is trying to place a food order."
    )
    wants_menu: bool = Field(
        default=False,
        description=(
            "True if the user asks to see the menu, dishes, prices "
            "or what is available."
        ),
    )
    items: list[OrderLine] = Field(
        default_factory=list,
        description="Every dish the user ordered, with its quantity."
    )


class OrderItem(TypedDict):
    dish: str
    quantity: int
    available: int
    price: int


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    items: list[OrderItem]
    status: str
    order_retry_attempts: int
    cook_retry_attempts: int
    serve_retry_attempts: int
    bill: dict
    payment: dict
    final_result: str


order_extraction_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are the order-extraction component of a restaurant food-ordering AI agent.

Your job is ONLY to determine whether the user is trying to place a food order
and to list every dish they ordered with its quantity.

A single message may contain one or many dishes.

Examples:
"I want 2 pizzas"                      -> pizza x2
"Give me 3 burgers and a pasta"        -> burger x3, pasta x1
"2 pizza, 1 burger, 2 biryani"         -> pizza x2, burger x1, biryani x2
"pasta 2 plates"                       -> pasta x2

Extract:
1. is_food_order
2. wants_menu
3. items: a list of (dish_name, quantity)

Set wants_menu = true (and is_food_order = false, items = []) when the user
asks to see the menu, prices or what is available, e.g.
"Show me the menu", "show me all menus", "What do you have?",
"What dishes are available?", "price list".

Normalize dish_name to lowercase and SINGULAR form ("pizzas" -> "pizza").
If a dish matches one of these menu items, use the exact menu name:
{menu_items}
If a dish is not on the menu, still return it as written (singular).
If no quantity is given for a dish, assume 1.

For unrelated questions such as:
"What is Python?"
"Tell me a joke."
"What is the weather?"
"Who are you?"

return:
is_food_order = false
items = []

Do not decide whether dishes are available. The Python
order-confirmation node will check the menu.
""",
    ),
    ("human", "{user_input}"),
])

order_extractor = order_extraction_prompt | llm.with_structured_output(
    OrderExtraction
)


def ai_message(text: str) -> AIMessage:
    return AIMessage(content=text)


def stock_of(dish: str) -> int:
    return MENU[dish]["stock"] if dish in MENU else 0


def normalize_dish(name: str) -> str:
    """Map LLM output like 'Pizzas' or 'burgers' onto a MENU key."""
    dish = name.lower().strip()
    if dish in MENU:
        return dish
    for suffix in ("es", "s"):
        if dish.endswith(suffix) and dish[: -len(suffix)] in MENU:
            return dish[: -len(suffix)]
    return dish


def describe_items(items: list[OrderItem]) -> str:
    return ", ".join(f"{i['quantity']} {i['dish']}" for i in items)


def format_menu() -> str:
    lines = ["=" * 50, "                 MENU", "=" * 50]
    for dish, info in MENU.items():
        stock = (
            f"Available: {info['stock']}" if info["stock"] > 0
            else "NOT AVAILABLE"
        )
        lines.append(f"{dish.title():<12} Rs {info['price']:<6} {stock}")
    lines.append("=" * 50)
    return "\n".join(lines)


def show_menu(state: State) -> dict:
    message = (
        "Please enter your order. You can order several dishes at once, "
        "for example: 2 pizzas and 1 burger."
    )
    return {
        "messages": [ai_message(message)],
        "status": "MENU_SHOWN",
    }


def get_order(state: State) -> dict:
    # Pause the graph until the user (CLI or web) supplies their order.
    user_input = interrupt({
        "type": "order",
        "prompt": "Enter one or more dishes with quantities.",
    })
    return {
        "messages": [HumanMessage(content=str(user_input))]
    }


def extract_order(state: State) -> dict:
    last_message = state["messages"][-1]

    result = order_extractor.invoke({
        "user_input": last_message.content,
        "menu_items": ", ".join(MENU),
    })

    if result.wants_menu and not result.items:
        return {
            "status": "MENU_REQUESTED",
            "messages": [ai_message(
                format_menu()
                + "\nWhat would you like? You can order several dishes at once."
            )],
        }

    if not result.is_food_order:
        message = (
            "I am a restaurant food-ordering AI agent. "
            "I can only help you place a food order. "
            "Please enter dishes and quantities."
        )
        return {
            "status": "UNRELATED",
            "messages": [ai_message(message)],
        }

    # Merge repeated dishes ("1 pizza and another pizza") into one line.
    merged: dict[str, int] = {}
    for line in result.items:
        if not line.dish_name or line.quantity <= 0:
            continue
        dish = normalize_dish(line.dish_name)
        merged[dish] = merged.get(dish, 0) + line.quantity

    if not merged:
        message = (
            "I could not understand the dishes or quantities. "
            "Please enter a valid order, e.g. 2 pizzas and 1 burger."
        )
        return {
            "status": "INVALID_ORDER",
            "messages": [ai_message(message)],
        }

    items: list[OrderItem] = [
        {
            "dish": dish,
            "quantity": quantity,
            "available": 0,
            "price": MENU[dish]["price"] if dish in MENU else 0,
        }
        for dish, quantity in merged.items()
    ]

    return {
        "items": items,
        "status": "ORDER_EXTRACTED",
    }


def route_after_extraction(state: State) -> Literal["get_order", "order_confirm"]:
    if state["status"] in ["UNRELATED", "INVALID_ORDER", "MENU_REQUESTED"]:
        return "get_order"
    return "order_confirm"


def order_confirm(state: State) -> dict:
    checked: list[OrderItem] = []
    lines = []

    for item in state["items"]:
        dish = item["dish"]
        available = min(stock_of(dish), item["quantity"])
        checked.append({**item, "available": available})

        if available == item["quantity"]:
            lines.append(f"- {item['quantity']} {dish}: available")
        elif available > 0:
            lines.append(
                f"- {item['quantity']} {dish}: only {available} available"
            )
        elif dish in MENU:
            lines.append(f"- {dish}: out of stock")
        else:
            lines.append(f"- {dish}: not on our menu")

    if all(i["available"] == i["quantity"] for i in checked):
        status = "CONFIRMED"
        header = "Everything you ordered is available:"
    elif any(i["available"] > 0 for i in checked):
        status = "PARTIAL"
        header = "Part of your order is available:"
    else:
        status = "NOT_AVAILABLE"
        header = "Sorry, none of those dishes are available:"

    remaining_attempts = state["order_retry_attempts"] - 1

    return {
        "items": checked,
        "status": status,
        "order_retry_attempts": remaining_attempts,
        "messages": [ai_message(header + "\n" + "\n".join(lines))],
    }


def route_after_order_confirm(
    state: State,
) -> Literal["cook", "partial_decision", "unavailable"]:
    if state["status"] == "CONFIRMED":
        return "cook"
    if state["status"] == "PARTIAL":
        return "partial_decision"
    return "unavailable"


def unavailable(state: State) -> dict:
    if state["order_retry_attempts"] <= 0:
        # end_failure posts the closing message; don't duplicate it here.
        message = "Sorry, you have used all 3 order attempts."
        return {
            "status": "FAILED",
            "messages": [ai_message(message)],
        }

    message = "Please choose other dishes from the menu."

    return {
        "status": "WAITING_FOR_NEW_ORDER",
        "messages": [ai_message(message)],
    }


def route_after_unavailable(
    state: State,
) -> Literal["show_menu", "end_failure"]:
    if state["order_retry_attempts"] <= 0:
        return "end_failure"
    return "show_menu"


ACCEPT_CHOICES = {"1", "yes", "y", "accept", "ok", "okay"}


def partial_decision(state: State) -> dict:
    adjusted = [
        {**i, "quantity": i["available"]}
        for i in state["items"] if i["available"] > 0
    ]

    choice = interrupt({
        "type": "partial",
        "prompt": (
            f"I can make: {describe_items(adjusted)}. "
            "1. Accept the adjusted order  "
            "2. Reject and place another order"
        ),
        "adjusted": [
            {"dish": i["dish"], "quantity": i["quantity"]} for i in adjusted
        ],
    })
    choice = str(choice).strip().lower()

    if choice in ACCEPT_CHOICES:
        message = (
            f"Okay. I will proceed with {describe_items(adjusted)}."
        )
        return {
            "items": adjusted,
            "status": "CONFIRMED",
            "messages": [HumanMessage(content="Accept"), ai_message(message)],
        }

    if state["order_retry_attempts"] <= 0:
        message = "Okay. The adjusted order has been rejected."
    else:
        message = (
            "Okay. The adjusted order has been rejected. "
            "Please place a new order."
        )

    return {
        "status": "ORDER_REJECTED",
        "messages": [HumanMessage(content="Reject"), ai_message(message)],
    }


def route_after_partial_decision(
    state: State,
) -> Literal["cook", "show_menu", "end_failure"]:
    if state["status"] == "CONFIRMED":
        return "cook"

    if state["order_retry_attempts"] <= 0:
        return "end_failure"

    return "show_menu"


# Set these to lists for deterministic testing.
# False = failure, True = success.
COOK_TEST_RESULTS = None
SERVE_TEST_RESULTS = None

cook_test_index = 0
serve_test_index = 0


def cook_operation() -> bool:
    global cook_test_index

    if COOK_TEST_RESULTS is not None:
        if cook_test_index >= len(COOK_TEST_RESULTS):
            return True
        result = COOK_TEST_RESULTS[cook_test_index]
        cook_test_index += 1
        return result

    return random.random() >= 0.40


def cook(state: State) -> dict:
    success = cook_operation()

    if success:
        message = (
            f"Cooking of {describe_items(state['items'])} "
            "completed successfully."
        )

        return {
            "status": "READY",
            "messages": [ai_message(message)],
        }

    remaining_retries = state["cook_retry_attempts"] - 1

    return {
        "status": "COOK_FAILED",
        "cook_retry_attempts": remaining_retries,
        "messages": [ai_message(
            "Unfortunately, cooking failed. "
            f"Cook retries remaining: {max(remaining_retries, 0)}."
        )],
    }


def route_after_cook(
    state: State,
) -> Literal["cook", "serve", "end_failure"]:
    if state["status"] == "READY":
        return "serve"

    if state["cook_retry_attempts"] >= 0:
        return "cook"

    return "end_failure"


def serve_operation() -> bool:
    global serve_test_index

    if SERVE_TEST_RESULTS is not None:
        if serve_test_index >= len(SERVE_TEST_RESULTS):
            return True
        result = SERVE_TEST_RESULTS[serve_test_index]
        serve_test_index += 1
        return result

    return random.random() >= 0.40


def serve(state: State) -> dict:
    success = serve_operation()

    if success:
        # Served food leaves the kitchen: update stock.
        for item in state["items"]:
            MENU[item["dish"]]["stock"] = max(
                stock_of(item["dish"]) - item["quantity"], 0
            )

        message = (
            f"Your order of {describe_items(state['items'])} "
            "has been served successfully."
        )

        return {
            "status": "SERVED",
            "messages": [ai_message(message)],
        }

    remaining_retries = state["serve_retry_attempts"] - 1

    return {
        "status": "SERVE_FAILED",
        "serve_retry_attempts": remaining_retries,
        "messages": [ai_message(
            "Unfortunately, serving failed. "
            f"Serve retries remaining: {max(remaining_retries, 0)}."
        )],
    }


def route_after_serve(
    state: State,
) -> Literal["retry_cook", "end_failure", "generate_bill"]:
    if state["status"] == "SERVED":
        return "generate_bill"

    if state["serve_retry_attempts"] < 0:
        return "end_failure"

    if state["cook_retry_attempts"] <= 0:
        return "end_failure"

    return "retry_cook"


def retry_cook(state: State) -> dict:
    remaining_retries = state["cook_retry_attempts"] - 1

    message = (
        "Serving failed. I will cook the order again "
        "before trying to serve it."
    )

    return {
        "status": "COOK_RETRY",
        "cook_retry_attempts": remaining_retries,
        "messages": [ai_message(message)],
    }


def generate_bill(state: State) -> dict:
    lines = [
        {
            "dish": item["dish"],
            "quantity": item["quantity"],
            "unit_price": item["price"],
            "amount": item["price"] * item["quantity"],
        }
        for item in state["items"]
    ]
    subtotal = sum(line["amount"] for line in lines)
    gst = round(subtotal * GST_RATE, 2)
    total = round(subtotal + gst, 2)

    bill = {
        "lines": lines,
        "subtotal": subtotal,
        "gst_rate": GST_RATE,
        "gst": gst,
        "total": total,
    }

    width = 36
    text = ["BILL", "-" * width]
    for line in lines:
        label = f"{line['quantity']} x {line['dish'].title()} @ {line['unit_price']}"
        text.append(f"{label:<26}{line['amount']:>10.2f}")
    text.append("-" * width)
    text.append(f"{'Subtotal':<26}{subtotal:>10.2f}")
    text.append(f"{f'GST {GST_RATE:.0%}':<26}{gst:>10.2f}")
    text.append(f"{'TOTAL (Rs)':<26}{total:>10.2f}")

    return {
        "bill": bill,
        "status": "BILLED",
        "messages": [
            ai_message("\n".join(text)),
            ai_message(
                f"Your total is Rs {total:.2f}. "
                "Please choose a payment method: UPI, Card or Cash."
            ),
        ],
    }


# Simulated payment: no card or bank details are collected.
PAYMENT_METHODS = {
    "1": "upi", "upi": "upi",
    "2": "card", "card": "card",
    "3": "cash", "cash": "cash",
}


def payment(state: State) -> dict:
    total = state["bill"]["total"]

    choice = interrupt({
        "type": "payment",
        "prompt": "Choose a payment method: 1. UPI  2. Card  3. Cash",
        "amount": total,
        "methods": ["upi", "card", "cash"],
    })
    method = PAYMENT_METHODS.get(str(choice).strip().lower())

    if not method:
        return {
            "status": "PAYMENT_PENDING",
            "messages": [
                HumanMessage(content=str(choice)),
                ai_message("Please choose UPI, Card or Cash to pay."),
            ],
        }

    payment_info = {
        "method": method,
        "amount": total,
        "transaction_id": "TXN" + uuid.uuid4().hex[:10].upper(),
    }
    message = (
        f"Payment of Rs {total:.2f} received via {method.upper()} "
        f"(transaction {payment_info['transaction_id']}). "
        "Thank you for dining with us!"
    )

    return {
        "payment": payment_info,
        "status": "PAID",
        "final_result": "ORDER_COMPLETED",
        "messages": [
            HumanMessage(content=f"Pay with {method.upper()}"),
            ai_message(message),
        ],
    }


def route_after_payment(state: State) -> Literal["payment", "end_success"]:
    if state["status"] == "PAID":
        return "end_success"
    return "payment"


def end_failure(state: State) -> dict:
    message = (
        "I am sorry, but I was unable to complete your order."
    )

    return {
        "status": "FAILED",
        "final_result": "ORDER_FAILED",
        "messages": [ai_message(message)],
    }


# ============================================================
# BUILD GRAPH
# ============================================================

graph = StateGraph(State)

graph.add_node("show_menu", show_menu)
graph.add_node("get_order", get_order)
graph.add_node("extract_order", extract_order)
graph.add_node("order_confirm", order_confirm)
graph.add_node("unavailable", unavailable)
graph.add_node("partial_decision", partial_decision)
graph.add_node("cook", cook)
graph.add_node("serve", serve)
graph.add_node("retry_cook", retry_cook)
graph.add_node("generate_bill", generate_bill)
graph.add_node("payment", payment)
graph.add_node("end_failure", end_failure)

graph.add_edge(START, "show_menu")
graph.add_edge("show_menu", "get_order")
graph.add_edge("get_order", "extract_order")

graph.add_conditional_edges(
    "extract_order",
    route_after_extraction,
    {
        "get_order": "get_order",
        "order_confirm": "order_confirm",
    },
)

graph.add_conditional_edges(
    "order_confirm",
    route_after_order_confirm,
    {
        "cook": "cook",
        "partial_decision": "partial_decision",
        "unavailable": "unavailable",
    },
)

graph.add_conditional_edges(
    "unavailable",
    route_after_unavailable,
    {
        "show_menu": "show_menu",
        "end_failure": "end_failure",
    },
)

graph.add_conditional_edges(
    "partial_decision",
    route_after_partial_decision,
    {
        "cook": "cook",
        "show_menu": "show_menu",
        "end_failure": "end_failure",
    },
)

graph.add_conditional_edges(
    "cook",
    route_after_cook,
    {
        "cook": "cook",
        "serve": "serve",
        "end_failure": "end_failure",
    },
)

graph.add_conditional_edges(
    "serve",
    route_after_serve,
    {
        "retry_cook": "retry_cook",
        "end_failure": "end_failure",
        "generate_bill": "generate_bill",
    },
)

graph.add_edge("retry_cook", "cook")
graph.add_edge("generate_bill", "payment")

graph.add_conditional_edges(
    "payment",
    route_after_payment,
    {
        "payment": "payment",
        "end_success": END,
    },
)
graph.add_edge("end_failure", END)

# The checkpointer lets the graph pause at interrupt() and resume later,
# which is what allows both the CLI and the web UI to drive it.
app = graph.compile(checkpointer=MemorySaver())


# ============================================================
# INITIAL STATE
# ============================================================

def new_initial_state() -> State:
    return {
        "messages": [],
        "items": [],
        "status": "START",

        # Maximum 3 actual order attempts
        "order_retry_attempts": 3,

        # Two additional cook retries
        "cook_retry_attempts": 2,

        # Two additional serve attempts (each re-cooks first)
        "serve_retry_attempts": 2,

        "bill": {},
        "payment": {},
        "final_result": "",
    }


def run_config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }


# ============================================================
# RUN (CLI)
# ============================================================

def run_cli() -> None:
    print("\n")
    print("=" * 60)
    print("RESTAURANT ORDER MANAGEMENT AI AGENT")
    print("=" * 60)

    config = run_config(str(uuid.uuid4()))
    graph_input: State | Command = new_initial_state()

    while True:
        for update in app.stream(graph_input, config, stream_mode="updates"):
            for node, values in update.items():
                if node == "__interrupt__":
                    continue
                if node == "show_menu":
                    print("\n" + format_menu())
                if node in ("cook", "serve", "retry_cook", "generate_bill"):
                    print(f"\n--- {node.upper()} NODE ---")
                for msg in (values or {}).get("messages", []):
                    if isinstance(msg, AIMessage):
                        print(f"\nAI: {msg.content}")

        snapshot = app.get_state(config)
        if not snapshot.next:
            break

        request = snapshot.tasks[0].interrupts[0].value
        if request["type"] == "partial":
            print(f"\nAI: {request['prompt'].split('. 1.')[0]}.")
            print("\nWhat would you like to do?")
            print("1. Accept the adjusted order")
            print("2. Reject and place another order")
        elif request["type"] == "payment":
            print("\nHow would you like to pay?")
            print("1. UPI")
            print("2. Card")
            print("3. Cash")

        graph_input = Command(resume=input("\nUser: "))

    print("\n")
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(app.get_state(config).values["final_result"])


if __name__ == "__main__":
    run_cli()
