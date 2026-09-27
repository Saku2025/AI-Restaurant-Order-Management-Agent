import os
import random
from typing import Annotated, TypedDict, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=GROQ_API_KEY,
)

MENU = {
    "pizza": 5,
    "burger": 10,
    "pasta": 3,
    "biryani": 0,
}


class OrderExtraction(BaseModel):
    is_food_order: bool = Field(
        description="True if the user is trying to place a food order."
    )
    dish_name: str | None = Field(
        default=None,
        description="Requested dish name, normalized to lowercase."
    )
    required_quantity: int | None = Field(
        default=None,
        description="Requested quantity."
    )


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    dish_name: str
    required_quantity: int
    available_quantity: int
    status: str
    order_retry_attempts: int
    cook_retry_attempts: int
    serve_retry_attempts: int
    final_result: str


order_extraction_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are the order-extraction component of a restaurant food-ordering AI agent.

Your job is ONLY to determine whether the user is trying to place a food order.

A food order contains dish name + quantity.

Examples:
"I want 2 pizzas"
"Give me 3 burgers"
"I need one pasta"
"pasta 2 plates"
"I want 5 biryani"

Extract:
1. is_food_order
2. dish_name
3. required_quantity

Normalize dish_name to lowercase.

For unrelated questions such as:
"What is Python?"
"Tell me a joke."
"What is the weather?"
"Who are you?"

return:
is_food_order = false
dish_name = null
required_quantity = null

Do not decide whether the dish is available. The Python
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


def show_menu(state: State) -> dict:
    print("\n")
    print("=" * 50)
    print("                 MENU")
    print("=" * 50)

    for dish, quantity in MENU.items():
        if quantity > 0:
            print(f"{dish.title():<15} Available: {quantity}")
        else:
            print(f"{dish.title():<15} NOT AVAILABLE")

    print("=" * 50)

    message = (
        "Please enter your order using the dish name and required quantity."
    )
    print(f"\nAI: {message}")

    return {
        "messages": [ai_message(message)],
        "status": "MENU_SHOWN",
    }


def get_order(state: State) -> dict:
    user_input = input("\nUser: ")
    return {
        "messages": [HumanMessage(content=user_input)]
    }


def extract_order(state: State) -> dict:
    last_message = state["messages"][-1]

    result = order_extractor.invoke({
        "user_input": last_message.content
    })

    if not result.is_food_order:
        message = (
            "I am a restaurant food-ordering AI agent. "
            "I can only help you place a food order. "
            "Please enter a dish and quantity."
        )
        print(f"\nAI: {message}")
        return {
            "status": "UNRELATED",
            "messages": [ai_message(message)],
        }

    if (
        not result.dish_name
        or not result.required_quantity
        or result.required_quantity <= 0
    ):
        message = (
            "I could not understand the dish or quantity. "
            "Please enter a valid dish and quantity."
        )
        print(f"\nAI: {message}")
        return {
            "status": "INVALID_ORDER",
            "messages": [ai_message(message)],
        }

    dish = result.dish_name.lower().strip()
    quantity = result.required_quantity

    print(
        f"\nExtracted Order:"
        f"\nDish     : {dish}"
        f"\nQuantity : {quantity}"
    )

    return {
        "dish_name": dish,
        "required_quantity": quantity,
        "status": "ORDER_EXTRACTED",
    }


def route_after_extraction(state: State) -> Literal["get_order", "order_confirm"]:
    if state["status"] in ["UNRELATED", "INVALID_ORDER"]:
        return "get_order"
    return "order_confirm"


def order_confirm(state: State) -> dict:
    dish = state["dish_name"]
    required_quantity = state["required_quantity"]

    available_quantity = MENU.get(dish, 0)

    if available_quantity >= required_quantity:
        status = "CONFIRMED"
        message = f"{required_quantity} {dish}(s) are available."
    elif available_quantity > 0:
        status = "PARTIAL"
        message = (
            f"You requested {required_quantity} {dish}(s), "
            f"but only {available_quantity} are available."
        )
    else:
        status = "NOT_AVAILABLE"
        message = f"Sorry, {dish} is not available."

    remaining_attempts = state["order_retry_attempts"] - 1

    print(f"\nAI: {message}")
    print(f"Order attempts remaining: {remaining_attempts}")

    return {
        "available_quantity": available_quantity,
        "status": status,
        "order_retry_attempts": remaining_attempts,
        "messages": [ai_message(message)],
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
        message = (
            "Sorry, you have used all 3 order attempts. "
            "The order process is ending."
        )
        print(f"\nAI: {message}")
        return {
            "status": "FAILED",
            "final_result": "ORDER_FAILED",
            "messages": [ai_message(message)],
        }

    message = (
        "This item is not available. "
        "Please choose another dish from the menu."
    )
    print(f"\nAI: {message}")

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


def partial_decision(state: State) -> dict:
    print("\nAI: The requested quantity is only partially available.")
    print(f"Available quantity: {state['available_quantity']}")
    print("\nWhat would you like to do?")
    print("1. Accept the available quantity")
    print("2. Reject and place another order")

    choice = input("\nUser: ").strip()

    if choice == "1":
        actual_quantity = state["available_quantity"]
        message = (
            f"Okay. I will proceed with "
            f"{actual_quantity} {state['dish_name']}(s)."
        )
        print(f"\nAI: {message}")

        return {
            "required_quantity": actual_quantity,
            "status": "CONFIRMED",
            "messages": [ai_message(message)],
        }

    message = (
        "Okay. The partial order has been rejected. "
        "Please choose another dish."
    )
    print(f"\nAI: {message}")

    return {
        "status": "ORDER_REJECTED",
        "messages": [ai_message(message)],
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
    print("\n--- COOK NODE ---")

    success = cook_operation()

    if success:
        message = (
            f"Cooking of {state['required_quantity']} "
            f"{state['dish_name']}(s) completed successfully."
        )
        print(f"AI: {message}")

        return {
            "status": "READY",
            "messages": [ai_message(message)],
        }

    remaining_retries = state["cook_retry_attempts"] - 1

    print("AI: Cooking failed.")
    print(f"Cook retries remaining: {remaining_retries}")

    return {
        "status": "COOK_FAILED",
        "cook_retry_attempts": remaining_retries,
        "messages": [ai_message("Unfortunately, cooking failed.")],
    }


def route_after_cook(
    state: State,
) -> Literal["cook", "serve", "end_failure"]:
    if state["status"] == "READY":
        return "serve"

    if state["cook_retry_attempts"] > 0:
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
    print("\n--- SERVE NODE ---")

    success = serve_operation()

    if success:
        message = (
            f"Your order of {state['required_quantity']} "
            f"{state['dish_name']}(s) has been served successfully."
        )
        print(f"\nAI: {message}")

        return {
            "status": "COMPLETE",
            "final_result": "ORDER_COMPLETED",
            "messages": [ai_message(message)],
        }

    remaining_retries = state["serve_retry_attempts"] - 1

    print("AI: Serving failed.")
    print(f"Serve retries remaining: {remaining_retries}")

    return {
        "status": "SERVE_FAILED",
        "serve_retry_attempts": remaining_retries,
        "messages": [ai_message("Unfortunately, serving failed.")],
    }


def route_after_serve(
    state: State,
) -> Literal["retry_cook", "end_failure", "end_success"]:
    if state["status"] == "COMPLETE":
        return "end_success"

    if state["serve_retry_attempts"] <= 0:
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

    print(f"\nAI: {message}")
    print(
        f"Cook retries remaining after this retry: "
        f"{remaining_retries}"
    )

    return {
        "status": "COOK_RETRY",
        "cook_retry_attempts": remaining_retries,
        "messages": [ai_message(message)],
    }


def route_after_retry_cook(
    state: State,
) -> Literal["cook", "end_failure"]:
    if state["cook_retry_attempts"] < 0:
        return "end_failure"
    return "cook"


def end_failure(state: State) -> dict:
    message = (
        "I am sorry, but I was unable to complete your order."
    )
    print(f"\nAI: {message}")

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
        "end_success": END,
    },
)

graph.add_conditional_edges(
    "retry_cook",
    route_after_retry_cook,
    {
        "cook": "cook",
        "end_failure": "end_failure",
    },
)

graph.add_edge("end_failure", END)

app = graph.compile()


# ============================================================
# INITIAL STATE
# ============================================================

initial_state: State = {
    "messages": [],
    "dish_name": "",
    "required_quantity": 0,
    "available_quantity": 0,
    "status": "START",

    # Maximum 3 actual order attempts
    "order_retry_attempts": 3,

    # Two additional cook retries
    "cook_retry_attempts": 2,

    # Two serve failure/retry cycles
    "serve_retry_attempts": 2,

    "final_result": "",
}


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("RESTAURANT ORDER MANAGEMENT AI AGENT")
    print("=" * 60)

    result = app.invoke(initial_state)

    print("\n")
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print(result["final_result"])