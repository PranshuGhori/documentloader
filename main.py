"""
Terminal chat — streaming answers, source table, conversation memory.

Run:
    python main.py            (/clear resets memory, /exit quits)
"""

import sys

from langchain_core.messages import AIMessage, HumanMessage
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag import chain, config


def sources_table(docs) -> Table:
    table = Table(show_header=True, header_style="bold", box=None)
    for col in ("#", "File", "Page", "Excerpt"):
        table.add_column(col)
    for s in chain.format_sources(docs, excerpt=70):
        table.add_row(str(s["index"]), s["source"], str(s["page"] or "-"), s["content"].replace("\n", " "))
    return table


def main() -> int:
    console = Console()
    try:
        stats = chain.vectorstore_stats()
        chain.default_llm()
    except config.ConfigError as e:
        console.print(f"[red]{e}[/red]")
        return 1

    console.print(
        Panel(
            f"RAG Q&A  ·  powered by {stats['model']}\n"
            f"{stats['chunks']} chunks indexed · top-{stats['retrieval_k']} retrieval",
            expand=False,
        )
    )

    history = []
    while True:
        try:
            question = console.input("\n[bold]You:[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question in ("/exit", "/quit"):
            break
        if question == "/clear":
            history = []
            console.print("[dim]Conversation cleared.[/dim]")
            continue

        try:
            docs = chain.retrieve(question)
            console.print("\n[bold]Assistant[/bold]")
            parts = []
            for token in chain.stream_answer(question, docs, history):
                parts.append(token)
                console.print(token, end="", markup=False, highlight=False)
            console.print()
        except chain.EmptyStoreError as e:
            console.print(f"[red]{e}[/red]")
            return 1
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted.[/dim]")
            continue
        except Exception as e:
            console.print(f"[red]Request failed: {e}[/red]")
            continue

        console.print(sources_table(docs))
        history = chain.trim_history(history + [HumanMessage(question), AIMessage("".join(parts))])
    return 0


if __name__ == "__main__":
    sys.exit(main())
