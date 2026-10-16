"""Command-line interface: ``georsagent <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .metrics import (
    DEFAULT_EQUIVALENCES,
    DEFAULT_TABLE,
    check_table_consistency,
    evaluate,
    load_equivalences,
    load_manuscript_table,
    load_raw_outputs,
)
from .profiles import build_profiles, write_prevalence_csv, write_profiles_markdown
from .taxonomy import KB_DIR, load_taxonomy
from .validator import STRICT_SCHEMA_PATH, build_strict_schema, load_schema, summarize_issues, validate_response

# Values reported in the manuscript (Abstract, Sec. 5.5, Sec. 7): k out of 10.
MANUSCRIPT_VALUES = {"exact_match": 8, "semantic_agreement": 9, "full_taxonomy_compliance": 3}


def _fmt_measure(name: str, m: dict) -> str:
    lo, hi = m["wilson95"]
    return f"  {name:<28} {m['k']:>2}/{m['n']:<2}  ({m['proportion']:.0%}; Wilson 95% CI {lo:.0%}-{hi:.0%})"


def _print_report(result: dict, title: str) -> None:
    print(title)
    print("-" * len(title))
    header = f"  {'test':<5} {'reference':<16} {'primary label':<16} {'exact':<6} {'semantic':<9} {'compliant':<9}"
    print(header)
    for row in result["per_test"]:
        comp = {True: "yes", False: "no", None: "?"}[row["fully_compliant"]]
        if row["compliance_source"] == "reported":
            comp += "*"
        print(f"  {row['test_id']:<5} {row['reference_class']:<16} {row['primary']:<16} "
              f"{'yes' if row['exact_match'] else 'no':<6} {'yes' if row['semantic_agreement'] else 'no':<9} {comp:<9}")
    print()
    print(_fmt_measure("Exact-match accuracy", result["exact_match"]))
    print(_fmt_measure("Semantic class agreement", result["semantic_agreement"]))
    print(_fmt_measure("Full taxonomy compliance", result["full_taxonomy_compliance"]))
    print(_fmt_measure("Primary label canonical", result["primary_label_canonical"]))
    if any(r["compliance_source"] == "reported" for r in result["per_test"]):
        print("\n  * compliance as reported in the manuscript (full alternative lists not available)")
    print("  Wilson intervals are descriptive additions, not reported in the manuscript.")


def cmd_metrics(args: argparse.Namespace) -> int:
    taxonomy = load_taxonomy(args.kb)
    equivalences = load_equivalences(args.equivalences)
    table_records = load_manuscript_table(args.table)
    status = 0

    problems = check_table_consistency(table_records, taxonomy)
    for p in problems:
        print(f"CONSISTENCY ERROR: {p}", file=sys.stderr)
    status |= bool(problems)

    table_result = evaluate(table_records, taxonomy, equivalences)
    _print_report(table_result, f"Image-based evaluation — from {Path(args.table).name}")
    for key, expected in MANUSCRIPT_VALUES.items():
        got = table_result[key]["k"]
        flag = "OK" if got == expected else "MISMATCH"
        print(f"  check vs manuscript: {key} = {got} (manuscript: {expected}) {flag}")
        status |= got != expected

    output = {"table": table_result}
    raw_files = sorted(Path(args.raw_dir).glob("*.json")) if args.raw_dir else []
    if args.raw_dir and not raw_files:
        print(f"\nNo raw outputs found in {args.raw_dir}; metrics above rely on the manuscript table.")
    if raw_files:
        raw_result = evaluate(load_raw_outputs(args.raw_dir), taxonomy, equivalences)
        print()
        _print_report(raw_result, f"Image-based evaluation — recomputed from {len(raw_files)} raw outputs")
        for key in MANUSCRIPT_VALUES:
            a, b = table_result[key]["k"], raw_result[key]["k"]
            if a != b:
                print(f"  MISMATCH raw vs table: {key} = {b} vs {a}", file=sys.stderr)
                status = 1
        output["raw"] = raw_result

    if args.json:
        Path(args.json).write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")
    return int(status)


def cmd_validate(args: argparse.Namespace) -> int:
    taxonomy = load_taxonomy(args.kb)
    schema = load_schema()
    worst = 0
    for file in args.files:
        doc = json.loads(Path(file).read_text(encoding="utf-8"))
        response = doc.get("response", doc)  # accept wrapped examples/raw outputs or bare responses
        issues = validate_response(response, taxonomy, schema)
        counts = summarize_issues(issues)
        verdict = "FAIL" if counts["error"] else "PASS"
        print(f"{verdict}  {file}  ({counts['error']} errors, {counts['warning']} warnings, {counts['info']} notes)")
        for issue in issues:
            if issue.level != "info" or args.verbose:
                print(f"    [{issue.level}] {issue.code} @ {issue.path}: {issue.message}")
        worst = max(worst, 1 if counts["error"] else 0)
    return worst


def cmd_build_profiles(args: argparse.Namespace) -> int:
    taxonomy = load_taxonomy(args.kb)
    exclude = set()
    if args.exclude:
        exclude = {line.strip() for line in Path(args.exclude).read_text(encoding="utf-8").splitlines() if line.strip()}
    if not args.split_column and not exclude:
        print("WARNING: no --split-column or --exclude given; profiles will include every annotated image, "
              "including any later used for testing.", file=sys.stderr)
    table = build_profiles(
        args.annotations, taxonomy,
        class_column=args.class_column, path_column=args.path_column,
        split_column=args.split_column, train_value=args.train_value,
        exclude=exclude, delimiter=args.sep,
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_prevalence_csv(table, out / "class_attribute_prevalence.csv")
    note = f"{Path(args.annotations).name}"
    if args.split_column:
        note += f", rows with {args.split_column} = {args.train_value}"
    if exclude:
        note += f", {len(exclude)} images excluded"
    write_profiles_markdown(table, taxonomy, out / "class_profiles.md",
                            typical_threshold=args.typical, distinctive_margin=args.distinctive, source_note=note)
    print(f"Profiles for {len(table.classes)} classes ({sum(table.counts.values())} images) written to {out}/")
    return 0


def cmd_export_strict_schema(args: argparse.Namespace) -> int:
    schema = build_strict_schema(load_taxonomy(args.kb))
    Path(args.out).write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="georsagent", description="GeoRS-Agent evaluation and audit tools")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--kb", default=str(KB_DIR), help="knowledge base directory")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("metrics", help="compute exact match, semantic agreement and taxonomy compliance")
    p.add_argument("--table", default=str(DEFAULT_TABLE))
    p.add_argument("--equivalences", default=str(DEFAULT_EQUIVALENCES))
    p.add_argument("--raw-dir", help="directory of raw agent outputs (one JSON per test)")
    p.add_argument("--json", help="write results to this JSON file")
    p.set_defaults(func=cmd_metrics)

    p = sub.add_parser("validate", help="audit structured responses against schema and taxonomy")
    p.add_argument("files", nargs="+")
    p.add_argument("-v", "--verbose", action="store_true", help="also show informational notes")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("build-profiles", help="derive class-attribute profiles from ABZSL annotations")
    p.add_argument("annotations")
    p.add_argument("--class-column", help="column with the scene class (default: parent folder of --path-column)")
    p.add_argument("--path-column", default="Filepath")
    p.add_argument("--split-column")
    p.add_argument("--train-value", default="train")
    p.add_argument("--exclude", help="text file listing image files to leave out (e.g. evaluation images)")
    p.add_argument("--sep", help="CSV delimiter (default: auto-detect)")
    p.add_argument("--typical", type=float, default=0.5)
    p.add_argument("--distinctive", type=float, default=0.3)
    p.add_argument("--out-dir", default=str(KB_DIR / "generated"))
    p.set_defaults(func=cmd_build_profiles)

    p = sub.add_parser("export-strict-schema", help="regenerate agent/output_schema.strict.json")
    p.add_argument("--out", default=str(STRICT_SCHEMA_PATH))
    p.set_defaults(func=cmd_export_strict_schema)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
