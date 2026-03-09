# import OpenAI from openai
import json

from openai import OpenAI
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# function 1: get_exchange_rate(currency_pair: str)
# Data:
# "USD_TWD" -> "32.0"
# "JPY_TWD" -> "0.2"
# "EUR_USD" -> "1.2"
# Return: A JSON string, e.g., {"currency_pair": "USD_TWD", "rate": "32.5"}
def get_exchange_rate(currency_pair: str) -> str:
    exchange_data = {
        "USD_TWD": "32.0",
        "JPY_TWD": "0.2",
        "EUR_USD": "1.2"
    }
    if currency_pair in exchange_data:
        return json.dumps({
            "currency_pair": currency_pair,
            "rate": exchange_data[currency_pair]
        })

    return json.dumps({
        "error": "Data not found"
    })

# function 2: get_stock_price(symbol: str)
# Data:
# "AAPL" -> "260.00"
# "TSLA" -> "430.00"
# "NVDA" -> "190.00"
# Return
# A JSON string, e.g., {"symbol": "AAPL", "price": "260.00"}.
def get_stock_price(symbol: str) -> str:
    stock_data = {
        "AAPL": "260.00",
        "TSLA": "430.00",
        "NVDA": "190.00"
    }
    if symbol in stock_data:
        return json.dumps({
            "symbol": symbol,
            "price": stock_data[symbol]
        })

    return json.dumps({
        "error": "Data not found"
    })

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the stock price for a given stock symbol",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock symbol like AAPL, TSLA, NVDA"
                    }
                },
                "required": ["symbol"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "Get exchange rate for a currency pair",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "currency_pair": {
                        "type": "string",
                        "description": "Currency pair like USD_TWD or EUR_USD"
                    }
                },
                "required": ["currency_pair"],
                "additionalProperties": False
            }
        }
    }
]

available_functions = {
    "get_stock_price": get_stock_price,
    "get_exchange_rate": get_exchange_rate
}

def fallback_parallel_tools(user_input: str) -> bool:
    text = user_input.upper()

    stock_symbols = ["AAPL", "TSLA", "NVDA"]
    currency_pairs = ["USD_TWD", "JPY_TWD", "EUR_USD"]

    found_items = []

    for symbol in stock_symbols:
        if symbol in text:
            found_items.append(("stock", symbol))

    for pair in currency_pairs:
        if pair in text:
            found_items.append(("exchange", pair))

    # 去掉重複
    unique_items = []
    seen = set()
    for item in found_items:
        if item not in seen:
            unique_items.append(item)
            seen.add(item)

    # 只在同一輪偵測到兩個以上項目時啟動 fallback
    if len(unique_items) < 2:
        return False

    results = []

    for item_type, value in unique_items:
        if item_type == "stock":
            print(f"[DEBUG] Tool called: get_stock_price({{'symbol': '{value}'}})")
            result = get_stock_price(value)
            data = json.loads(result)
            results.append(data)

        elif item_type == "exchange":
            print(f"[DEBUG] Tool called: get_exchange_rate({{'currency_pair': '{value}'}})")
            result = get_exchange_rate(value)
            data = json.loads(result)
            results.append(data)

    answer_parts = []

    for data in results:
        if "error" in data:
            answer_parts.append("Data not found")
        elif "symbol" in data:
            answer_parts.append(f"The price of {data['symbol']} is ${data['price']}")
        elif "currency_pair" in data:
            answer_parts.append(f"The exchange rate of {data['currency_pair']} is {data['rate']}")

    assistant_reply = "Assistant: " + ", and ".join(answer_parts) + "."
    print(assistant_reply)

    messages.append({
        "role": "assistant",
        "content": assistant_reply
    })

    return True

messages = [
    {"role": "system", "content": "You are a Financial Assistant. You can only help with stock prices, exchange rates, and simple conversation memory. Use the provided tools when needed. Do not claim to have other tools. If data is not found, say 'Data not found' politely."}
]

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    messages.append({"role": "user", "content": user_input})
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
    except Exception as e:
        print(f"[DEBUG ERROR] {e}")

        if fallback_parallel_tools(user_input):
            continue

        print("Assistant: Sorry, the model failed to call the tool correctly. Please try again.")
        messages.pop()
        continue

    assistant_message = response.choices[0].message
    messages.append(assistant_message)

    if assistant_message.tool_calls:
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"[DEBUG] Tool called: {function_name}({function_args})")

            function_to_call = available_functions[function_name]
            function_response = function_to_call(**function_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": function_response
            })

        try:
            second_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
        except Exception as e:
            print("Assistant: Sorry, there was a tool-calling error.")
            print(f"[DEBUG ERROR] {e}")
            continue

        final_message = second_response.choices[0].message
        messages.append(final_message)
        print(final_message.content)
    else:
        print(assistant_message.content)
        