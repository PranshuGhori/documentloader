"""
Retrieval eval: does the expected page appear in the top-k chunks?

    python -m eval.run_eval                     # offline hashing embedder
    python -m eval.run_eval --embedder openai   # needs OPENAI_API_KEY

Builds a throwaway index of the sample document; never touches ./chroma_db.
"""

import argparse
import json
import tempfile
from pathlib import Path

from rag import chain, config
from rag.embeddings import HashingEmbeddings, openai_embeddings
from rag.ingest import build_index, load_documents, split_documents

HERE = Path(__file__).parent


def evaluate(embeddings, document: str, questions: list[dict], k: int) -> tuple[dict, list[int | None]]:
    with tempfile.TemporaryDirectory() as tmp:
        vs = chain.open_vectorstore(embeddings, persist_dir=tmp)
        build_index(split_documents(load_documents(document)), vs)
        ranks = []
        for q in questions:
            pages = [chain.format_sources([d])[0]["page"] for d in chain.retrieve(q["question"], vs, k=k)]
            ranks.append(pages.index(q["page"]) + 1 if q["page"] in pages else None)
    n = len(ranks)
    metrics = {
        "hit@1": sum(r == 1 for r in ranks) / n,
        f"hit@{k}": sum(r is not None for r in ranks) / n,
        f"mrr@{k}": sum(1 / r for r in ranks if r) / n,
    }
    return metrics, ranks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedder", choices=["hashing", "openai"], default="hashing")
    parser.add_argument("--k", type=int, default=config.RETRIEVAL_K)
    parser.add_argument("--document", default="documents/ps1.pdf")
    parser.add_argument("--questions", default=str(HERE / "questions.jsonl"))
    parser.add_argument("-v", "--verbose", action="store_true", help="print the rank for each question")
    args = parser.parse_args()

    questions = [json.loads(line) for line in Path(args.questions).read_text().splitlines() if line.strip()]
    embeddings = HashingEmbeddings() if args.embedder == "hashing" else openai_embeddings()
    metrics, ranks = evaluate(embeddings, args.document, questions, args.k)

    if args.verbose:
        for q, r in zip(questions, ranks, strict=True):
            print(f"  {r or '-':>2}  p{q['page']}  {q['question']}")
    print(
        f"embedder={args.embedder} questions={len(questions)} "
        f"chunk_size={config.CHUNK_SIZE} overlap={config.CHUNK_OVERLAP} k={args.k}"
    )
    print("  ".join(f"{name}={value:.3f}" for name, value in metrics.items()))


if __name__ == "__main__":
    main()
