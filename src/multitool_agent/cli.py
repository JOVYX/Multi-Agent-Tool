"""Interactive Command Line Interface for the Multi-Tool AI Agent."""

from __future__ import annotations

import logging
import sys
from typing import Optional

from multitool_agent.agent import Agent
from multitool_agent.config import load_settings
from multitool_agent.llm import MissingAPIKeyError, create_llm_client
from multitool_agent.models import ToolCall, ToolResult
from multitool_agent.utils.logging import configure_logging, get_logger

logger = get_logger("multitool_agent.cli")

BANNER = """\
╭──────────────────────────────────────────╮
│           Multi-Tool AI Agent            │
│   Powered by Groq (console.groq.com)     │
╰──────────────────────────────────────────╯

Type '/help' for commands.
Type '/exit' or '/quit' to leave.
"""

HELP_TEXT = """\
Available Commands:
  /help, help     - Show this help message
  /tools, tools   - List all available tools and their capabilities
  /clear, clear   - Clear conversation history
  /status         - Show current configuration and workspace status
  /exit, /quit    - Exit the agent session

Examples to try:
  • "What is 125 * 17.5 / 100?"
  • "Search for the latest Python release and its release date."
  • "Calculate 15% of 2000 and save the result to workspace/result.txt"
  • "Read workspace/notes.txt and summarize the important points."
"""


def _tool_start_callback(tc: ToolCall) -> None:
    """Print clean indicator when a tool begins executing."""
    args_preview = ", ".join(f"{k}={repr(v)}" for k, v in tc.arguments.items())
    if len(args_preview) > 60:
        args_preview = args_preview[:57] + "..."
    print(f"\n  ⚡ Running tool: {tc.name}({args_preview})")


def _tool_end_callback(tr: ToolResult) -> None:
    """Print clean indicator when a tool completes."""
    preview = tr.content.strip().split("\n")[0]
    if len(preview) > 70:
        preview = preview[:67] + "..."
    if tr.is_error:
        print(f"  ✗ {tr.content.strip()}")
    else:
        print(f"  ✓ Result: {preview}")


def run_cli() -> None:
    """Run the main interactive CLI loop."""
    settings = load_settings()
    configure_logging(level=settings.log_level)

    # In interactive CLI, keep prompt clean unless DEBUG is explicitly requested
    if settings.log_level != "DEBUG":
        logging.getLogger("multitool_agent").setLevel(logging.WARNING)


    print(BANNER)

    # Validate active API key
    if not settings.has_active_llm_key:
        if settings.llm_provider == "groq":
            print(
                "⚠️  NOTE: GROQ_API_KEY is not configured.\n"
                "   Create a .env file from .env.example and set your API key:\n"
                "     cp .env.example .env\n"
                "     # Add your GROQ_API_KEY from https://console.groq.com\n"
                "   (You can still run /help, /tools, or exit)\n"
            )
        else:
            print(
                "⚠️  NOTE: ANTHROPIC_API_KEY is not configured.\n"
                "   Create a .env file from .env.example and set your ANTHROPIC_API_KEY.\n"
                "   (You can still run /help, /tools, or exit)\n"
            )

    llm = create_llm_client(settings)
    agent = Agent(
        settings=settings,
        llm_client=llm,
        on_tool_start=_tool_start_callback,
        on_tool_end=_tool_end_callback,
    )

    while True:
        try:
            user_input = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        normalized_cmd = user_input.lower()

        # Command handling
        if normalized_cmd in {"/exit", "/quit", "exit", "quit"}:
            print("Goodbye!")
            break

        elif normalized_cmd in {"/help", "help"}:
            print(HELP_TEXT)
            continue

        elif normalized_cmd in {"/clear", "clear"}:
            agent.clear_history()
            print("✓ Conversation history cleared.\n")
            continue

        elif normalized_cmd in {"/tools", "tools"}:
            print("\nAvailable tools:\n")
            for tool in agent.registry.list_tools():
                print(f"• {tool.name}")
                print(f"  Description: {tool.description}")
                schema_props = tool.schema().get("input_schema", {}).get("properties", {})
                if schema_props:
                    params = ", ".join(f"{k} ({v.get('type', 'any')})" for k, v in schema_props.items())
                    print(f"  Parameters:  {params}")
                print()
            continue

        elif normalized_cmd in {"/status"}:
            summary = settings.masked_summary()
            print("\nCurrent Configuration:")
            for k, v in summary.items():
                print(f"  {k:24}: {v}")
            print(f"  conversation_turns      : {len(agent.conversation_history)}")
            print()
            continue

        # Process user prompt via agent with real-time streaming
        try:
            print()
            first_token_printed = [False]

            def stream_callback(token: str) -> None:
                if not first_token_printed[0]:
                    print("Agent > ", end="", flush=True)
                    first_token_printed[0] = True
                print(token, end="", flush=True)

            response = agent.run(user_input, stream=True, stream_callback=stream_callback)
            if not first_token_printed[0]:
                print(f"Agent > {response.text}")
            print()
        except MissingAPIKeyError as e:
            print(f"\nConfiguration Error: {str(e)}\n")
        except Exception as e:
            logger.exception("Error during agent execution")
            print(f"\nError: {type(e).__name__}: {str(e)}\n")



def main() -> None:
    """Entry point for the console script and python -m."""
    try:
        run_cli()
    except KeyboardInterrupt:
        print("\nSession terminated.")
        sys.exit(0)


if __name__ == "__main__":
    main()
