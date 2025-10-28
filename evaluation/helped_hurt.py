"""
Surface queries where RAG helped vs hurt compared to baseline using /qa endpoint.
Writes top 10 helped and top 10 hurt examples to results/helped_hurt.json.
"""

import json
import requests


def score_len(s: str) -> float:
    return len(s or "")


def main():
    samples = [
        "What is in front of the Notre Dame Main Building?",
        "Who first visited New York Harbor in 1524?",
        "What city became the last capital under the Articles of Confederation?",
        "What is machine learning?",
    ]
    rows = []
    for q in samples:
        r = requests.post("http://127.0.0.1:8000/qa", json={"question": q}).json()
        delta = score_len(r.get("after")) - score_len(r.get("before"))
        rows.append({"q": q, "before": r.get("before"), "after": r.get("after"), "delta": delta})

    helped = sorted([x for x in rows if x["delta"] > 0], key=lambda x: -x["delta"])[:10]
    hurt = sorted([x for x in rows if x["delta"] <= 0], key=lambda x: x["delta"])[:10]

    out = {"helped": helped, "hurt": hurt}
    with open("results/helped_hurt.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote results/helped_hurt.json")


if __name__ == "__main__":
    main()


