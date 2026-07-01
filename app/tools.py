"""
Tool Calling
============
Implements the two mock tools required by the spec:

1. get_order_status(order_id) -> reads data/orders.json
2. search_product(name)       -> reads data/products.json with simple
   category-aware matching so queries like "laptops" do not return
   accessory products such as "Laptop Stand".

Each tool has:
- a plain Python function that does the actual lookup
- an OpenAI "function calling" JSON schema describing it to the LLM

The LLM decides *when* to call these based on the schema descriptions and
the conversation; app/llm.py executes whichever one the LLM asks for.

Data is loaded fresh on every call (files are tiny) so edits to the JSON
files are picked up without restarting the server.
"""
import json
import re
from typing import Optional
from app.config import settings


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_CATEGORY_ALIASES = {
    "laptop": {"laptop", "laptops"},
    "mouse": {"mouse", "mice"},
    "keyboard": {"keyboard", "keyboards"},
    "monitor": {"monitor", "monitors"},
    "hub": {"hub", "hubs"},
    "stand": {"stand", "stands"},
}
_ACCESSORY_ONLY_TERMS = {
    "laptop": {"stand", "sleeve", "bag", "case", "dock", "holder"},
    "mouse": {"pad", "mat"},
    "keyboard": {"cover", "case"},
}


def _load_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text.lower()))


def _detect_categories(query: str) -> list[str]:
    """Return all product categories mentioned in the query (e.g. laptops + keyboards)."""
    query_tokens = _tokenize(query)
    categories: list[str] = []
    for category, aliases in _CATEGORY_ALIASES.items():
        if query_tokens & aliases:
            categories.append(category)
    return categories


def _detect_category(query: str) -> Optional[str]:
    categories = _detect_categories(query)
    return categories[0] if categories else None


def _matches_category(
    product: dict, category: str, query_tokens: set[str]
) -> bool:
    product_name = product["name"]
    product_tokens = _product_tokens(product_name)
    aliases = _CATEGORY_ALIASES[category]
    if not (product_tokens & aliases):
        return False

    accessory_terms = _ACCESSORY_ONLY_TERMS.get(category, set())
    if (
        accessory_terms
        and (product_tokens & accessory_terms)
        and not (query_tokens & accessory_terms)
    ):
        return False
    return True


def _search_category(products: list[dict], category: str, query_tokens: set[str]) -> list[dict]:
    matches = [
        p for p in products
        if _matches_category(p, category, query_tokens)
    ]
    matches.sort(key=lambda p: (p["price"], p["name"]))
    return matches


def _product_tokens(product_name: str) -> set[str]:
    return _tokenize(product_name)


# ---------------------------------------------------------------------------
# Tool 1: Order Status
# ---------------------------------------------------------------------------
def get_order_status(order_id: str) -> dict:
    """Look up an order by ID in orders.json."""
    orders = _load_json(settings.ORDERS_FILE)
    order_id_norm = order_id.strip().upper()

    for order in orders:
        if order["order_id"].upper() == order_id_norm:
            return {
                "found": True,
                "order_id": order["order_id"],
                "status": order["status"],
                "estimated_delivery": order.get("estimated_delivery"),
            }

    return {
        "found": False,
        "order_id": order_id,
        "message": f"No order found with ID '{order_id}'.",
    }


# ---------------------------------------------------------------------------
# Tool 2: Product Search
# ---------------------------------------------------------------------------
def search_product(name: str) -> dict:
    """
    Category-aware product search over products.json.

    Supports multiple categories in one query (e.g. "laptops and keyboards").
    If the query clearly refers to a category like laptops or mice, we only
    return products that match that category and we avoid accessory matches
    such as "Laptop Stand" when the user asked for laptops.
    """
    products = _load_json(settings.PRODUCTS_FILE)
    query = name.strip().lower()
    query_tokens = _tokenize(query)
    categories = _detect_categories(query)

    if len(categories) > 1:
        groups = []
        all_matches: list[dict] = []
        for category in categories:
            matches = _search_category(products, category, query_tokens)
            if not matches:
                continue
            results = [
                {
                    "name": p["name"],
                    "price": p["price"],
                    "in_stock": p["stock"] > 0,
                    "stock": p["stock"],
                }
                for p in matches
            ]
            groups.append({"category": category, "results": results})
            all_matches.extend(matches)

        if not groups:
            return {
                "found": False,
                "query": name,
                "message": f"No products found matching '{name}'.",
            }

        return {
            "found": True,
            "query": name,
            "groups": groups,
            "results": [
                {
                    "name": p["name"],
                    "price": p["price"],
                    "in_stock": p["stock"] > 0,
                    "stock": p["stock"],
                }
                for p in all_matches
            ],
        }

    category = categories[0] if categories else None
    matches = []
    for product in products:
        if category:
            if _matches_category(product, category, query_tokens):
                matches.append(product)
            continue

        if query and query in product["name"].lower():
            matches.append(product)

    if not matches:
        return {
            "found": False,
            "query": name,
            "message": f"No products found matching '{name}'.",
        }

    matches.sort(key=lambda p: (p["price"], p["name"]))

    return {
        "found": True,
        "query": name,
        "results": [
            {
                "name": p["name"],
                "price": p["price"],
                "in_stock": p["stock"] > 0,
                "stock": p["stock"],
            }
            for p in matches
        ],
    }


def search_products_cheaper_than(max_price: float, category_hint: Optional[str] = None) -> dict:
    """
    Helper used when the user says something like "cheaper options" after
    already discussing a product/category. Not exposed directly as an LLM
    tool -- app/llm.py's system prompt instructs the model to call
    search_product again with a lower implied price filtered client-side,
    OR (simpler, and what we actually do) the LLM just re-reads the prior
    search_product results already in conversation history and reasons
    over them. This function is kept as a convenience utility for that
    reasoning step / for unit tests.
    """
    products = _load_json(settings.PRODUCTS_FILE)
    query = (category_hint or "").strip().lower()

    candidates = [
        p for p in products
        if (not query or query in p["name"].lower()) and p["price"] < max_price
    ]
    candidates.sort(key=lambda p: p["price"])

    return {
        "found": len(candidates) > 0,
        "max_price": max_price,
        "results": candidates,
    }


# ---------------------------------------------------------------------------
# OpenAI function-calling schemas
# ---------------------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": (
                "Get the status and estimated delivery date of an order, "
                "given its order ID (e.g. 'ORD001'). Use this whenever the "
                "user asks about the status, location, or delivery date of "
                "an order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID, e.g. 'ORD001'",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_product",
            "description": (
                "Search the product catalog by name (substring match) and "
                "return price and stock availability. Use this whenever the "
                "user asks if a product is available, its price, or wants "
                "to browse/compare products (e.g. 'do you have a wireless "
                "mouse', 'show me laptops', 'cheaper options')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Product name or keyword to search for, e.g. "
                            "'laptop', 'wireless mouse', 'keyboard'"
                        ),
                    }
                },
                "required": ["name"],
            },
        },
    },
]

# Dispatch table: tool name -> Python callable
TOOL_DISPATCH = {
    "get_order_status": lambda args: get_order_status(args["order_id"]),
    "search_product": lambda args: search_product(args["name"]),
}