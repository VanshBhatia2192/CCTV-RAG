"""Summarise the relevance judgments collected through the demo (outputs/feedback/judgments.jsonl).
Only results someone actually marked are counted: precision here is 'of the results you judged, how many were right', not a full
retrieval evaluation (unjudged results are unknown). The same query judged twice keeps the latest label."""
import json, sys
from collections import defaultdict
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/feedback/judgments.jsonl")
if not path.exists():
    sys.exit(f"no judgments yet ({path})")
last = {}
for line in path.read_text().splitlines():
    if line.strip():
        r = json.loads(line); last[(r["type"], r["kind"], r["mode"], r["query"], r["id"])] = r
groups = defaultdict(list)
for r in last.values():
    groups[(r["type"], r["kind"], r["mode"])].append(r)
print(f"{len(last)} judged results, {len({(r['type'], r['query']) for r in last.values()})} distinct queries\n")
print(f"{'type':7s} {'kind':9s} {'mode':10s} {'judged':>6s} {'right':>6s} {'prec':>6s} {'prec@5':>7s} {'prec@10':>8s}  queries")
for (t, k, m), rs in sorted(groups.items(), key=lambda x: str(x[0])):
    def prec(sub):
        return f"{100 * sum(r['label'] for r in sub) / len(sub):5.0f}%" if sub else "   –  "
    top5 = [r for r in rs if (r.get("rank") or 99) <= 5]; top10 = [r for r in rs if (r.get("rank") or 99) <= 10]
    print(f"{t:7s} {k:9s} {str(m):10s} {len(rs):6d} {sum(r['label'] for r in rs):6d} {prec(rs)} {prec(top5):>7s} {prec(top10):>8s}  {len({r['query'] for r in rs})}")
