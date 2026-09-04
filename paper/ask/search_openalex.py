#!/usr/bin/env python3
"""Search OpenAlex for prior work related to the GPB/OPB/EPB paper.

The API key is read from OPENALEX_API_KEY and is never written to output files.
The script intentionally searches broad neighboring literatures because the
paper's exact acronym is unlikely to be the terminology used by prior work.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_QUERIES = {
    "bottleneck_and_prior": [
        '"conditional entropy bottleneck"',
        '"conditional information bottleneck"',
        '"information bottleneck" "label conditional"',
        '"class conditional" prior variational representation',
        '"geometric prior" bottleneck',
        '"structured latent prior" variational',
        '"class conditional Gaussian prior" representation',
        '"prior geometry" information bottleneck',
        '"geometric information bottleneck"',
        '"geometric prior" "information bottleneck"',
        '"orthogonal information bottleneck"',
        '"prototype prior" bottleneck representation',
        '"isometric information bottleneck"',
        '"prior bottleneck" representation learning',
        '"geometric prior bottleneck"',
        '"label-conditioned prior" bottleneck',
        '"orthogonal prior" representation learning',
        '"equidistant prior" regression',
    ],
    "classification_geometry": [
        '"orthogonal prototypes" classifier',
        '"orthogonal classifier" adversarial robustness',
        '"orthonormal class means" representation learning',
        '"equiangular tight frame" classifier',
        '"neural collapse" variational',
        '"neural collapse" information bottleneck',
        '"hyperspherical prototypes" classifier',
        '"polar decomposition" orthogonal classifier',
        '"Stiefel manifold" classifier representation',
    ],
    "regression_geometry": [
        '"isometric embedding" regression representation',
        '"distance preserving" representation regression',
        '"metric preserving" supervised representation',
        '"geometric regression" latent representation',
        '"Lipschitz representation" regression',
        '"label embedding" regression geometry',
        '"equidistant" representation regression',
    ],
    "robustness": [
        '"information bottleneck" adversarial robustness',
        '"conditional entropy bottleneck" adversarial',
        '"compressed representation" adversarial robustness',
        '"stochastic latent" adversarial robustness',
        '"representation geometry" adversarial robustness',
        '"orthogonal classifier" adversarial examples',
    ],
}

TITLE_TERMS = {
    "conditional entropy bottleneck": 12,
    "conditional information bottleneck": 10,
    "information bottleneck": 7,
    "variational bottleneck": 6,
    "geometric prior": 10,
    "structured prior": 8,
    "label conditional": 8,
    "class conditional": 7,
    "orthogonal": 8,
    "orthonormal": 8,
    "prototype": 6,
    "class mean": 7,
    "equiangular": 8,
    "tight frame": 8,
    "neural collapse": 7,
    "stiefel": 7,
    "polar": 5,
    "isometric": 8,
    "distance preserving": 8,
    "metric preserving": 8,
    "geometric regression": 7,
    "adversarial robustness": 5,
    "adversarial": 3,
}


def reconstruct_abstract(work: dict[str, Any]) -> str:
    inverted = work.get("abstract_inverted_index") or {}
    positions: list[tuple[int, str]] = []
    for word, indexes in inverted.items():
        for index in indexes:
            positions.append((index, word))
    positions.sort()
    return compact(" ".join(word for _, word in positions))


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize(text: str) -> str:
    return compact(text).lower()


def score_work(work: dict[str, Any]) -> tuple[int, list[str]]:
    title = normalize(work.get("title", ""))
    abstract = normalize(reconstruct_abstract(work))
    keywords = normalize(" ".join(k.get("display_name", "") for k in work.get("keywords", [])))
    text = f"{title} {keywords} {abstract}"
    score = 0
    hits: list[str] = []
    for term, weight in TITLE_TERMS.items():
        if term in title:
            score += weight * 3
            hits.append(f"title:{term}")
        elif term in keywords:
            score += weight * 2
            hits.append(f"keyword:{term}")
        elif term in text:
            score += weight
            hits.append(term)
    return score, hits


def api_get(query: str, api_key: str, mailto: str | None, per_page: int) -> dict[str, Any]:
    params = {
        "search": query,
        "per-page": str(per_page),
        # Relevance is more appropriate for candidate discovery than citation
        # count, which otherwise floods broad queries with unrelated classics.
        "sort": "relevance_score:desc",
        "api_key": api_key,
        "select": ",".join(
            [
                "id",
                "doi",
                "title",
                "publication_year",
                "type",
                "cited_by_count",
                "authorships",
                "primary_location",
                "open_access",
                "abstract_inverted_index",
                "keywords",
            ]
        ),
    }
    if mailto:
        params["mailto"] = mailto
    url = "https://api.openalex.org/works?" + urlencode(params)
    request = Request(url, headers={"User-Agent": "IB-paper-literature-survey/1.0"})
    with urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if "results" not in payload:
        raise RuntimeError(f"OpenAlex response did not contain results for query {query!r}")
    return payload


def author_string(work: dict[str, Any]) -> str:
    names = []
    for authorship in work.get("authorships", [])[:8]:
        author = authorship.get("author") or {}
        if author.get("display_name"):
            names.append(author["display_name"])
    if len(work.get("authorships", [])) > 8:
        names.append("et al.")
    return ", ".join(names)


def open_url(work: dict[str, Any]) -> str:
    location = work.get("primary_location") or {}
    landing = location.get("landing_page_url")
    if landing:
        return landing
    doi = work.get("doi")
    return doi or work.get("id", "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-query", type=int, default=25, help="results fetched per query")
    parser.add_argument("--top-per-group", type=int, default=15, help="works shown per query group")
    parser.add_argument("--mailto", default=None, help="optional OpenAlex polite-pool email")
    parser.add_argument("--sleep", type=float, default=0.15, help="delay between requests")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()

    api_key = os.environ.get("OPENALEX_API_KEY")
    if not api_key:
        print("Set OPENALEX_API_KEY before running this script.", file=sys.stderr)
        return 2
    if args.per_query < 1 or args.top_per_group < 1:
        print("--per-query and --top-per-group must be positive.", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_by_query: dict[str, Any] = {}
    works_by_id: dict[str, dict[str, Any]] = {}
    groups_by_id: defaultdict[str, set[str]] = defaultdict(set)

    for group, queries in DEFAULT_QUERIES.items():
        for query in queries:
            print(f"Querying {group}: {query}", file=sys.stderr)
            try:
                payload = api_get(query, api_key, args.mailto, args.per_query)
            except Exception as exc:  # keep completed queries usable
                print(f"  failed: {exc}", file=sys.stderr)
                continue
            raw_by_query[query] = payload
            for work in payload.get("results", []):
                work_id = work.get("id")
                if not work_id:
                    continue
                works_by_id[work_id] = work
                groups_by_id[work_id].add(group)
            time.sleep(max(args.sleep, 0.0))

    scored: list[dict[str, Any]] = []
    for work_id, work in works_by_id.items():
        score, hits = score_work(work)
        scored.append(
            {
                "id": work_id,
                "title": compact(work.get("title", "")),
                "year": work.get("publication_year"),
                "type": work.get("type", ""),
                "cited_by_count": work.get("cited_by_count", 0),
                "authors": author_string(work),
                "doi": work.get("doi") or "",
                "url": open_url(work),
                "groups": ";".join(sorted(groups_by_id[work_id])),
                "relevance_score": score,
                "matched_terms": ";".join(hits),
                "abstract": reconstruct_abstract(work),
            }
        )
    scored.sort(key=lambda row: (row["relevance_score"], row["cited_by_count"]), reverse=True)

    (output_dir / "openalex_raw.json").write_text(
        json.dumps(raw_by_query, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    (output_dir / "openalex_results.json").write_text(
        json.dumps(scored, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    with (output_dir / "openalex_results.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "id",
            "title",
            "year",
            "type",
            "cited_by_count",
            "authors",
            "doi",
            "url",
            "groups",
            "relevance_score",
            "matched_terms",
            "abstract",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scored)

    lines = [
        "# OpenAlex literature survey",
        "",
        "This report is a broad candidate survey for the GPB/OPB/EPB manuscript.",
        "Relevance scores are heuristic; every candidate must be read before making a novelty claim.",
        "",
        f"- Queries completed: {len(raw_by_query)} / {sum(len(v) for v in DEFAULT_QUERIES.values())}",
        f"- Unique works: {len(scored)}",
        "- API endpoint: `https://api.openalex.org/works`",
        "",
    ]
    for group, queries in DEFAULT_QUERIES.items():
        lines.extend([f"## {group}", ""])
        group_rows = [row for row in scored if group in row["groups"].split(";")]
        group_rows.sort(key=lambda row: (row["relevance_score"], row["cited_by_count"]), reverse=True)
        seen_titles: set[str] = set()
        shown = 0
        for row in group_rows:
            title_key = normalize(row["title"])
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            lines.append(
                f"- **{row['title']}** ({row['year'] or 'n.d.'}) -- {row['authors'] or 'authors unavailable'}; "
                f"cited {row['cited_by_count']}; score {row['relevance_score']}. "
                f"[OpenAlex record]({row['url']})"
            )
            shown += 1
            if shown >= args.top_per_group:
                break
        if not group_rows:
            lines.append("- No results returned.")
        lines.append("")

    (output_dir / "literature_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(scored)} works to {output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
