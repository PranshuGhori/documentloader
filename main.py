"""
Terminal Q&A interface — streaming output, source citations, conversation memory.

Run:
    python main.py
"""

import os
import sys
from langchain_core.messages import HumanMessage, AIMessage
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.prompt import Prompt
from rich import box
from rag import chain, config

console = Console()


def print_sources(docs: list) -> None:
    if not docs:
        return
    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold dim",
        title="[dim]Sources used[/dim]",
        title_justify="left",
        expand=False,
    )
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Page", justify="center", style="yellow", width=6)
    table.add_column("Excerpt", style="white")

    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        source = os.path.basename(meta.get("source", "—"))
        page = str(meta.get("page", 0) + 1) if meta.get("page") is not None else "—"
        excerpt = doc.page_content[:100].replace("\n", " ") + "…"
        table.add_row(str(i), source, page, excerpt)

    console.print(table)


def main() -> None:
    if not os.path.exists(config.PERSIST_DIR):
        console.print(
            "\n[bold red]Vector store not found.[/bold red] "
            "Run [bold]python create_database.py[/bold] first.\n"
        )
        sys.exit(1)

    try:
        stats = chain.vectorstore_stats()
    except Exception:
        stats = {}

    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]RAG Q&A[/bold cyan]  ·  powered by [bold]{config.MODEL}[/bold]\n"
            f"[dim]{stats.get('chunks', '?')} chunks indexed  ·  "
            f"top-{config.RETRIEVAL_K} retrieval  ·  type [bold]quit[/bold] to exit[/dim]",
            border_style="cyan",
            padding=(0, 2),
        )
    )

    history: list = []

    while True:
        console.print()
        try:
            question = Prompt.ask("[bold green]You[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break

        docs = chain.retrieve(question)

        console.print()
        console.print("[bold blue]Assistant[/bold blue]")

        full_answer = ""
        with Live(Text(""), console=console, refresh_per_second=20) as live:
            for token in chain.stream_answer(question, docs, history):
                full_answer += token
                live.update(Text(full_answer))

        console.print()
        print_sources(docs)

        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=full_answer))
        if len(history) > 20:
            history = history[-20:]


if __name__ == "__main__":
    main()
