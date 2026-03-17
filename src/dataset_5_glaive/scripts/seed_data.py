"""
Seed script for DS5 (Glaive) - generates synthetic JSONL for dummy mode.

Creates glaive_raw.jsonl with records matching the Glaive Function Calling v2
schema (system prompt, chat content with function calls/responses).

Usage:
    python -m src.dataset_5_glaive.scripts.seed_data
"""

import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

random.seed(42)

SYSTEM_PROMPTS = [
    "You are a helpful assistant with access to the following functions.",
    "You are an AI assistant that can call external tools to help users.",
    "You are a function-calling assistant. Use the provided tools when needed.",
]

FUNCTION_DEFS = [
    {"name": "get_weather", "description": "Get current weather for a location", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}},
    {"name": "search_web", "description": "Search the web for information", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}},
    {"name": "send_email", "description": "Send an email to a recipient", "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}}},
    {"name": "create_reminder", "description": "Create a reminder at a specific time", "parameters": {"type": "object", "properties": {"message": {"type": "string"}, "time": {"type": "string"}}}},
    {"name": "calculate", "description": "Perform a mathematical calculation", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}},
    {"name": "translate_text", "description": "Translate text between languages", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_lang": {"type": "string"}}}},
    {"name": "get_stock_price", "description": "Get current stock price", "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}}},
    {"name": "set_alarm", "description": "Set an alarm", "parameters": {"type": "object", "properties": {"time": {"type": "string"}, "label": {"type": "string"}}}},
]

USER_MESSAGES = [
    "What's the weather like in San Francisco?",
    "Search for the latest news about AI.",
    "Send an email to john@example.com about the meeting tomorrow.",
    "Remind me to call the dentist at 3pm.",
    "What is 15% of 250?",
    "Translate 'hello world' to French.",
    "What's the current price of AAPL stock?",
    "Set an alarm for 7:30 AM labeled 'morning workout'.",
    "Can you check the weather in Tokyo?",
    "Search for Python tutorial for beginners.",
    "Email sarah@example.com with subject 'Project Update'.",
    "What is the square root of 144?",
    "Translate 'good morning' to Spanish.",
    "Set an alarm for 6:00 AM.",
    "What's the stock price for GOOGL?",
]

FUNCTION_CALLS = [
    {"name": "get_weather", "arguments": '{"location": "San Francisco"}'},
    {"name": "search_web", "arguments": '{"query": "latest AI news"}'},
    {"name": "send_email", "arguments": '{"to": "john@example.com", "subject": "Meeting Tomorrow", "body": "Reminder about our meeting."}'},
    {"name": "create_reminder", "arguments": '{"message": "Call the dentist", "time": "3:00 PM"}'},
    {"name": "calculate", "arguments": '{"expression": "0.15 * 250"}'},
    {"name": "translate_text", "arguments": '{"text": "hello world", "target_lang": "fr"}'},
    {"name": "get_stock_price", "arguments": '{"symbol": "AAPL"}'},
    {"name": "set_alarm", "arguments": '{"time": "7:30 AM", "label": "morning workout"}'},
    {"name": "get_weather", "arguments": '{"location": "Tokyo"}'},
    {"name": "search_web", "arguments": '{"query": "Python tutorial beginners"}'},
    {"name": "send_email", "arguments": '{"to": "sarah@example.com", "subject": "Project Update", "body": "Here is the update."}'},
    {"name": "calculate", "arguments": '{"expression": "sqrt(144)"}'},
    {"name": "translate_text", "arguments": '{"text": "good morning", "target_lang": "es"}'},
    {"name": "set_alarm", "arguments": '{"time": "6:00 AM", "label": "wake up"}'},
    {"name": "get_stock_price", "arguments": '{"symbol": "GOOGL"}'},
]

FUNCTION_RESPONSES = [
    "The weather in San Francisco is 65°F, partly cloudy.",
    "Top results: 1. New AI model achieves breakthrough... 2. AI regulation updates...",
    "Email sent successfully to john@example.com.",
    "Reminder set for 3:00 PM: Call the dentist.",
    "37.5",
    "bonjour le monde",
    "AAPL: $178.52 (+1.2%)",
    "Alarm set for 7:30 AM - morning workout.",
    "The weather in Tokyo is 72°F, sunny.",
    "Top results: 1. Python.org tutorial... 2. W3Schools Python...",
    "Email sent successfully to sarah@example.com.",
    "12",
    "buenos días",
    "Alarm set for 6:00 AM - wake up.",
    "GOOGL: $142.30 (-0.5%)",
]

ASSISTANT_REPLIES = [
    "The current weather in San Francisco is 65°F and partly cloudy.",
    "Here are the latest AI news results I found for you.",
    "I've sent the email to john@example.com about the meeting tomorrow.",
    "Done! I've set a reminder for 3pm to call the dentist.",
    "15% of 250 is 37.5.",
    "'Hello world' in French is 'bonjour le monde'.",
    "The current price of AAPL is $178.52, up 1.2% today.",
    "Your alarm is set for 7:30 AM with the label 'morning workout'.",
    "The weather in Tokyo is 72°F and sunny.",
    "I found some great Python tutorials for beginners.",
    "Email sent to sarah@example.com with subject 'Project Update'.",
    "The square root of 144 is 12.",
    "'Good morning' in Spanish is 'buenos días'.",
    "Alarm set for 6:00 AM.",
    "GOOGL is currently at $142.30, down 0.5%.",
]


def _build_record(idx: int) -> dict:
    i = idx % len(USER_MESSAGES)
    funcs = random.sample(FUNCTION_DEFS, k=min(3, len(FUNCTION_DEFS)))
    system_prompt = random.choice(SYSTEM_PROMPTS)
    func_defs_str = json.dumps(funcs)

    system_content = f"{system_prompt}\n\nAvailable functions:\n{func_defs_str}"

    chat = (
        f"USER: {USER_MESSAGES[i]}\n"
        f"ASSISTANT: <functioncall> {json.dumps(FUNCTION_CALLS[i])}\n"
        f"FUNCTION RESPONSE: {FUNCTION_RESPONSES[i]}\n"
        f"ASSISTANT: {ASSISTANT_REPLIES[i]}"
    )

    return {
        "system": system_content,
        "chat": chat,
    }


def generate_all(output_file: Path, num_records: int = 100) -> int:
    """Generate synthetic Glaive-style JSONL records."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(num_records):
            record = _build_record(i)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"DS5 seed: wrote {num_records} records to {output_file}")
    return num_records


def main():
    try:
        from src.config.paths import get_ds5_raw_dir
        raw_dir = get_ds5_raw_dir()
    except ImportError:
        raw_dir = PROJECT_ROOT / "data" / "raw" / "ds5_glaive"

    output = raw_dir / "glaive_raw.jsonl"
    generate_all(output)
    return True


if __name__ == "__main__":
    main()
