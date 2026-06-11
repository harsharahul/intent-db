"""End-to-end IntentDB demo: one corpus, one query, three intents.

Run: python examples/demo.py
"""

import tempfile
from pathlib import Path

from intentdb import IntentDB

CORPUS = [
    ("py-lang", "Python is a programming language; developers write code, "
                "functions and modules, then debug their programs.", {"topic": "software"}),
    ("py-snake", "The python is a large snake, a reptile that lives in jungle "
                 "habitats and hunts wildlife at night.", {"topic": "nature"}),
    ("pip-doc", "Use pip to install python packages and manage code "
                "dependencies for programming projects.", {"topic": "software"}),
    ("monty", "Monty Python was a British comedy group famous for sketch "
              "humor, films and absurd jokes.", {"topic": "entertainment"}),
    ("zoo-doc", "At the zoo you can watch a snake exhibit in the reptile "
                "house, full of jungle animals.", {"topic": "nature"}),
    ("flask-doc", "Flask is a python web framework: write code for routes, "
                  "templates and APIs.", {"topic": "software"}),
]

INTENTS = {
    "coding": dict(
        description="software programming, source code, debugging programs",
        exemplars=["how do I write code", "install a package", "debug my program"],
    ),
    "wildlife": dict(
        description="animals, reptiles, snakes, jungle habitats and nature",
        exemplars=["what do snakes eat", "jungle animal habitats"],
    ),
    "comedy": dict(
        description="jokes, humor, comedy sketches, films and entertainment",
        exemplars=["funny sketch comedy", "famous comedians"],
    ),
}


def show(title, results):
    print(f"\n=== {title} ===")
    for i, r in enumerate(results, 1):
        text = r.text if len(r.text) <= 72 else r.text[:69] + "..."
        print(f"  {i}. [{r.score:+.4f}] ({r.doc_key}) {text}")


def main():
    path = Path(tempfile.mkdtemp()) / "demo.intentdb"
    with IntentDB(path) as db:
        db.add_many([(t, k, m) for k, t, m in CORPUS])
        for name, spec in INTENTS.items():
            db.register_intent(name, **spec)
        print(f"database: {db.stats()}")

        # The headline: same query, three intents, three different answers.
        for intent in INTENTS:
            show(f'query "python"  intent={intent}',
                 db.query("python", intent=intent, k=3))

        # No intent declared: IntentDB infers it from the query.
        hits = db.query("python sketch jokes", k=2)
        show(f'query "python sketch jokes"  (inferred intent: {hits[0].intent})', hits)

        # The classifier's view of an ambiguous vs. a clear query.
        print("\n=== explain ===")
        for q in ("python", "debug my python program"):
            print(f"  {q!r:32} -> {db.explain(q)['inferred_intent']}")


if __name__ == "__main__":
    main()
