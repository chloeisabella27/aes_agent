"""
AES Analysis Agent: OpenAI tool-calling loop with guardrails and safety limits.
Uses tools in agent/tools.py only (no subprocess). Loads .env from project root.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Optional

# Ensure project root is on path so "from agent.tools import ..." works from any CWD
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
from openai import OpenAI

from agent.tools import (
    list_experiments,
    resolve_experiment,
    predict_next_scan as tool_predict_next_scan,
    get_prediction_summary,
    plot_predicted_spectrum,
)

# Load .env from project root
load_dotenv(_PROJECT_ROOT / ".env")

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

ALLOWED_TOOLS = {
    "list_experiments",
    "resolve_experiment",
    "predict_next_scan",
    "get_prediction_summary",
    "plot_predicted_spectrum",
}
MAX_TURNS = 8
MAX_TOOL_CALLS = 3
TOOL_TIMEOUT_SECONDS = 30
GUARDRAIL_MESSAGE = "I'm sorry, I can't perform that task with the current AES analysis tools."

# Conversation state: last experiment for which predict_next_scan succeeded (used for "plot it" follow-up)
last_prediction_experiment: Optional[str] = None

PLOT_IT_PHRASES = [
    "plot it",
    "plot that",
    "plot the prediction",
    "show the plot",
    "visualize the prediction",
    "show the predicted spectrum",
    "yes please",
]

SYSTEM_PROMPT = """You are an AI assistant for Auger Electron Spectroscopy (AES) analysis.
You are connected to a machine learning pipeline that predicts the next Ti MVV Auger spectrum.

You may only perform these tasks:
1. List available experiments
2. Predict the next Ti MVV spectrum for an experiment
3. Explain prediction results (using only data returned from tools; never invent data)
4. Generate a visualization of a predicted spectrum when the user asks to plot or visualize the prediction.

You may generate a visualization of a predicted spectrum using plot_predicted_spectrum(experiment) when the user asks to plot or visualize the prediction.

Rules:
- Always use resolve_experiment when the user's experiment name might be misspelled or unclear.
- Only call predict_next_scan with an experiment name returned by list_experiments or resolve_experiment.
- Use only results returned from tools. Never invent RMSE values or file paths.
- For any unsupported task, respond exactly with: "I'm sorry, I can't perform that task with the current AES analysis tools."
"""

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "list_experiments",
            "description": "List all available experiment names (e.g. TF268, TF288) from the dataset.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_experiment",
            "description": "Resolve a possibly misspelled or partial experiment name to the best match and alternatives. Call this before predict_next_scan when the user's experiment name might be wrong.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "User-supplied experiment name or fragment (e.g. tf2688, TF268)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_next_scan",
            "description": "Run the ML pipeline to predict the next Ti MVV spectrum for the given experiment. Use an experiment name from list_experiments or resolve_experiment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment": {"type": "string", "description": "Experiment name (e.g. TF268)."},
                },
                "required": ["experiment"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_prediction_summary",
            "description": "Get the latest prediction summary (RMSE, paths, scan index) for an experiment, or the most recent run if no experiment is specified.",
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment": {"type": "string", "description": "Experiment name (optional). Omit for the most recent prediction."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_predicted_spectrum",
            "description": "Generate and save a PNG plot of the predicted Ti MVV spectrum for an experiment. Use this when the user asks to visualize or plot the prediction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment": {
                        "type": "string",
                        "description": "Experiment name such as TF268",
                    },
                },
                "required": ["experiment"],
            },
        },
    },
]


def _run_tool(name: str, arguments: dict) -> str:
    """Execute a single tool by name; return JSON string for the LLM. Only ALLOWED_TOOLS are run. Enforces TOOL_TIMEOUT_SECONDS."""
    if name not in ALLOWED_TOOLS:
        return json.dumps({"error": "tool not allowed", "tool": name})

    def _execute():
        if name == "list_experiments":
            return list_experiments()
        if name == "resolve_experiment":
            return resolve_experiment(arguments.get("query", ""))
        if name == "predict_next_scan":
            return tool_predict_next_scan(arguments.get("experiment", ""))
        if name == "get_prediction_summary":
            return get_prediction_summary(arguments.get("experiment"))
        if name == "plot_predicted_spectrum":
            return plot_predicted_spectrum(arguments.get("experiment", ""))
        return {"error": "unknown tool", "tool": name}

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_execute)
            result = fut.result(timeout=TOOL_TIMEOUT_SECONDS)
        return json.dumps(result)
    except FuturesTimeoutError:
        return json.dumps({"error": "The requested analysis timed out.", "tool": name})
    except Exception as e:
        return json.dumps({"error": str(e), "tool": name})


def _run_agent_turn(client: OpenAI, messages: list, tool_calls_count: int) -> tuple[list, str, int]:
    """
    Send messages to the API. If the model returns tool_calls, execute them (respecting ALLOWED_TOOLS and timeout),
    append results, and call again until the model returns a normal content reply or we hit limits.
    Returns (updated_messages, final_content, new_tool_calls_count).
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOL_DEFINITIONS,
        temperature=0.2,
        max_tokens=1024,
        top_p=0.9,
    )
    choice = response.choices[0]
    message = choice.message
    tool_calls = list(message.tool_calls or [])

    if not tool_calls:
        return messages + [{"role": "assistant", "content": message.content or ""}], (message.content or "").strip(), tool_calls_count

    if tool_calls_count + len(tool_calls) > MAX_TOOL_CALLS:
        return messages + [{"role": "assistant", "content": "Tool call limit reached."}], "Tool call limit reached.", tool_calls_count + len(tool_calls)

    # Build tool results and append to messages
    tool_messages = [{"role": "assistant", "content": message.content or "", "tool_calls": [
        {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
        for tc in tool_calls
    ]}]
    for tc in tool_calls:
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except json.JSONDecodeError:
            args = {}
        result = _run_tool(name, args)
        if name == "predict_next_scan":
            try:
                data = json.loads(result)
                if "error" not in data and data.get("experiment"):
                    global last_prediction_experiment
                    last_prediction_experiment = args.get("experiment") or data.get("experiment")
            except json.JSONDecodeError:
                pass
        tool_messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result,
        })
    new_messages = messages + tool_messages
    return _run_agent_turn(client, new_messages, tool_calls_count + len(tool_calls))


def run_agent() -> None:
    if not API_KEY:
        print("Error: OPENAI_API_KEY is not set. Add it to .env at the project root and try again.")
        return
    client = OpenAI(api_key=API_KEY)
    print("\nAES Analysis Agent Ready (OpenAI tool-calling)")
    print("Type 'quit' to exit.\n")
    turn = 0
    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() == "quit":
            break
        turn += 1
        if turn > MAX_TURNS:
            print("\nAgent: Maximum interaction depth reached.\n")
            continue

        msg = user_input.strip().lower()
        if any(phrase in msg for phrase in PLOT_IT_PHRASES):
            if last_prediction_experiment is not None:
                result = _run_tool("plot_predicted_spectrum", {"experiment": last_prediction_experiment})
                try:
                    data = json.loads(result)
                    if data.get("plot_file"):
                        print(f"\nAgent: Plot saved to: {data['plot_file']}\n")
                    elif data.get("error"):
                        print(f"\nAgent: {data['error']}\n")
                    else:
                        print(f"\nAgent: {result}\n")
                except json.JSONDecodeError:
                    print(f"\nAgent: {result}\n")
            else:
                print("\nAgent: No prediction has been generated yet. Please run a prediction first.\n")
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]
        try:
            final_messages, content, _ = _run_agent_turn(client, messages, 0)
            print(f"\nAgent: {content}\n")
        except Exception as e:
            print(f"\nAgent: Error: {e}\n")


if __name__ == "__main__":
    run_agent()
