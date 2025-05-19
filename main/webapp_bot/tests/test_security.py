import subprocess, pathlib, sys

BAD_WORDS = ["eval(", "assert False", "password ="]

def test_static_scan():
    root = pathlib.Path(__file__).parent.parent
    py_files = list(root.rglob("*.py"))
    bad = [f for f in py_files if any(b in f.read_text() for b in BAD_WORDS)]
    assert not bad, f"Insecure patterns: {bad}"

def test_bandit_clean():
    res = subprocess.run(
        ["bandit", "-q", "-r", "server_bot.py"], capture_output=True, text=True
    )
    assert ">> Issue:" not in res.stdout
