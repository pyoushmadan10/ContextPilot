"""Allow running contextpilot as a module: python -m contextpilot

Dispatches to subcommands:
  python -m contextpilot serve <project_root>  — start MCP server
  python -m contextpilot scan <project_root>   — scan without server
  python -m contextpilot stats <project_root>  — show index stats
"""

import sys


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python -m contextpilot <command> [args...]\n"
            "\n"
            "Commands:\n"
            "  serve <project_root>  Start the MCP server\n"
            "  scan  <project_root>  Scan and index (no server)\n"
            "  stats <project_root>  Show index statistics\n",
            file=sys.stderr,
        )
        sys.exit(1)

    command = sys.argv[1]
    # Shift argv so subcommands see their own args
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "serve":
        from contextpilot.server import main as serve_main
        serve_main()

    elif command == "scan":
        from contextpilot.scanner import scan_project
        import json

        if len(sys.argv) < 2:
            print("Usage: python -m contextpilot scan <project_root>", file=sys.stderr)
            sys.exit(1)

        result = scan_project(sys.argv[1])
        print(json.dumps({
            "files_scanned": result.files_scanned,
            "files_indexed": result.files_indexed,
            "files_skipped": result.files_skipped,
            "files_deleted": result.files_deleted,
            "symbols_total": result.symbols_total,
            "duration_seconds": round(result.duration_seconds, 2),
        }, indent=2))

    elif command == "stats":
        from contextpilot.store import VectorStore
        import json

        if len(sys.argv) < 2:
            print("Usage: python -m contextpilot stats <project_root>", file=sys.stderr)
            sys.exit(1)

        with VectorStore(sys.argv[1]) as store:
            stats = store.get_stats()
            print(json.dumps(stats, indent=2))

    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Run 'python -m contextpilot' for usage.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
