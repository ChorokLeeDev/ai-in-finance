#!/usr/bin/env python
"""
MoE-RAG Command Line Interface

Usage:
    moe-rag train ./docs/ -o model.pt
    moe-rag retrieve model.pt "your question here"

That's it. No configuration needed.
"""

import argparse
import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cmd_train(args):
    """Train MoE-RAG on documents."""
    from moe_rag import MoERAG

    print("=" * 50)
    print("MoE-RAG Training")
    print("=" * 50)

    # Load documents
    if os.path.isdir(args.input):
        rag = MoERAG.from_directory(args.input, device=args.device)
    elif os.path.isfile(args.input):
        rag = MoERAG.from_files([args.input], device=args.device)
    else:
        print(f"Error: {args.input} not found")
        sys.exit(1)

    # Train
    rag.train(epochs=args.epochs, verbose=True)

    # Save
    rag.save(args.output)

    print("\nDone! Use 'moe-rag retrieve' to query.")


def cmd_retrieve(args):
    """Retrieve relevant passages."""
    from moe_rag import MoERAG

    # Load model
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    rag = MoERAG.load(args.model, device=args.device)

    # Query
    results = rag.retrieve(args.query, top_k=args.top_k)

    # Display results
    print()
    for i, result in enumerate(results):
        print(f"[{result.score:.2f}] {result.text[:200]}...")
        if result.source:
            print(f"       Source: {result.source}")
        print()


def cmd_interactive(args):
    """Interactive query mode."""
    from moe_rag import MoERAG

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    print("Loading model...")
    rag = MoERAG.load(args.model, device=args.device)

    print("\nMoE-RAG Interactive Mode")
    print("Type your questions. Ctrl+C to exit.\n")

    while True:
        try:
            query = input("Query: ").strip()
            if not query:
                continue

            results = rag.retrieve(query, top_k=args.top_k)

            print()
            for result in results:
                print(f"[{result.score:.2f}] {result.text[:200]}...")
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="MoE-RAG: Replace RAG pipelines with learned attention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train on a directory of documents
    moe-rag train ./my_docs/ -o my_rag.pt

    # Train on a single file
    moe-rag train ./document.txt -o my_rag.pt

    # Query the trained model
    moe-rag retrieve my_rag.pt "How do I reset my password?"

    # Interactive mode
    moe-rag interactive my_rag.pt
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train on documents")
    train_parser.add_argument("input", help="Directory or file with documents")
    train_parser.add_argument("-o", "--output", default="moe_rag.pt",
                             help="Output model path (default: moe_rag.pt)")
    train_parser.add_argument("--epochs", type=int, default=10,
                             help="Training epochs (default: 10)")
    train_parser.add_argument("--device", default="cpu",
                             help="Device (cpu/cuda)")

    # Retrieve command
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve relevant passages")
    retrieve_parser.add_argument("model", help="Path to trained model")
    retrieve_parser.add_argument("query", help="Query string")
    retrieve_parser.add_argument("-k", "--top-k", type=int, default=3,
                                help="Number of results (default: 3)")
    retrieve_parser.add_argument("--device", default="cpu",
                                help="Device (cpu/cuda)")

    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive query mode")
    interactive_parser.add_argument("model", help="Path to trained model")
    interactive_parser.add_argument("-k", "--top-k", type=int, default=3,
                                   help="Number of results (default: 3)")
    interactive_parser.add_argument("--device", default="cpu",
                                   help="Device (cpu/cuda)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "train":
        cmd_train(args)
    elif args.command == "retrieve":
        cmd_retrieve(args)
    elif args.command == "interactive":
        cmd_interactive(args)


if __name__ == "__main__":
    main()
