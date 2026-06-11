import json

import pytest

from intentdb.cli import main


@pytest.fixture()
def db_path(tmp_path):
    return str(tmp_path / "cli.intentdb")


def run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr().out
    return code, out


def test_init_and_stats(capsys, db_path):
    code, out = run(capsys, "init", db_path, "--embedder", "hashing:dim=256")
    assert code == 0
    stats = json.loads(out)
    assert stats["documents"] == 0
    assert stats["dim"] == 256


def test_full_workflow(capsys, db_path):
    run(capsys, "init", db_path)
    code, out = run(
        capsys, "add", db_path,
        "Python is a programming language for writing code.",
        "--key", "py", "--metadata", '{"topic": "software"}',
    )
    assert code == 0 and out.strip() == "py"

    run(capsys, "add", db_path, "The python snake is a jungle reptile.", "--key", "snake")

    code, _ = run(
        capsys, "intent", "add", db_path, "coding",
        "--description", "software programming and source code",
        "--exemplar", "write code", "--exemplar", "debug a program",
    )
    assert code == 0

    code, out = run(capsys, "query", db_path, "python", "--intent", "coding", "--json")
    assert code == 0
    hits = json.loads(out)
    assert hits[0]["doc_key"] == "py"
    assert hits[0]["intent"] == "coding"

    code, out = run(capsys, "explain", db_path, "debug my program code")
    assert json.loads(out)["inferred_intent"] == "coding"

    code, out = run(capsys, "intent", "list", db_path)
    assert [i["name"] for i in json.loads(out)] == ["coding"]

    code, out = run(capsys, "delete", db_path, "snake")
    assert code == 0 and "deleted" in out

    code, out = run(capsys, "stats", db_path)
    assert json.loads(out)["documents"] == 1


def test_add_file_split_paragraphs(capsys, db_path, tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("first paragraph here\n\nsecond paragraph here\n")
    run(capsys, "init", db_path)
    code, out = run(capsys, "add", db_path, "--file", str(notes), "--split-paragraphs")
    assert code == 0
    assert len(out.strip().splitlines()) == 2

    code, out = run(capsys, "stats", db_path)
    assert json.loads(out)["documents"] == 2


def test_add_nothing_errors(capsys, db_path):
    run(capsys, "init", db_path)
    code = main(["add", db_path])
    assert code == 2


def test_human_readable_query_output(capsys, db_path):
    run(capsys, "init", db_path)
    run(capsys, "add", db_path, "alpha beta gamma", "--key", "a")
    code, out = run(capsys, "query", db_path, "alpha")
    assert code == 0
    assert "a: alpha beta gamma" in out
