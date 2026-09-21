"""Isolate candidate rollouts from the live module-global TraCI connection."""

import base64
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import numpy as np
from .results import write_json


class SimulatorOracle:
    def __init__(self, directory, source):
        self.directory = Path(directory)
        private = self.directory / "source"
        private.mkdir(parents=True, exist_ok=False)
        for name in ("Environment", "Data", "rl_att"):
            shutil.copytree(str(source / name), str(private / name), ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("oarl.py", "main.py", "requirements.txt"):
            shutil.copy2(str(source / name), str(private / name))
        self.hashes = {p.relative_to(private).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in private.rglob("*") if p.is_file()}
        self.private = private
        self.command = [sys.executable, "-u", "-m", "rl_att.evaluation.oracle_worker"]
        self.log = (self.directory / "stderr.log").open("w", encoding="utf-8")
        self.process = subprocess.Popen(self.command, cwd=str(private), stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.log, universal_newlines=True,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        self.status = "running"

    def request(self, command, **fields):
        self.process.stdin.write(json.dumps(dict(command=command, **fields), allow_nan=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            self.status = "failed"
            raise RuntimeError("Simulator oracle exited; inspect " + str(self.directory / "stderr.log"))
        try:
            reply = json.loads(line)
        except ValueError:
            self.status = "failed"
            raise RuntimeError("Invalid oracle IPC response: " + repr(line))
        if not reply["ok"]:
            self.status = "failed"
            raise RuntimeError(reply["error"])
        return reply["result"]

    def reset(self, snapshot, observation):
        return self.request("reset", snapshot=base64.b64encode(snapshot).decode("ascii"),
                            initial=dict(observation=np.asarray(observation).tolist(), reward=0.0,
                                         done=False, collision=False))

    @staticmethod
    def snapshot(env):
        return pickle.dumps((env.__dict__, np.random.get_state()), protocol=4)

    def begin(self):
        return self.request("begin")

    def step(self, action):
        return self.request("step", action=int(action))

    def budgeted_step(self, action, remaining):
        return self.request("budgeted_step", action=int(action), remaining=remaining)

    def observe_fallback(self, action, observation, reward, done, collision):
        return self.request("observe_fallback", action=int(action), transition=dict(
            observation=np.asarray(observation).tolist(), reward=float(reward), done=bool(done), collision=bool(collision)))

    def observe(self, action, observation, reward, done, collision):
        return self.request("observe", action=int(action), transition=dict(
            observation=np.asarray(observation).tolist(), reward=float(reward), done=bool(done), collision=bool(collision)))

    def counts(self):
        return self.request("counts")

    def close(self):
        counts = None
        try:
            if self.process.poll() is None and self.status == "running":
                counts = self.request("close")
                self.process.wait(timeout=30)
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=30)
            if self.status == "running":
                self.status = "passed" if self.process.returncode == 0 else "failed"
        finally:
            self.log.close()
            after = {name: hashlib.sha256((self.private / name).read_bytes()).hexdigest() for name in self.hashes}
            changed = [name for name in after if after[name] != self.hashes[name]]
            if any(name != "Data/StraightRoad.sumocfg" for name in changed):
                self.status = "failed"
            write_json(self.directory / "manifest.json", dict(command=self.command, cwd=str(self.private),
                       status=self.status, returncode=self.process.poll(), counts=counts,
                       source_sha256_before=self.hashes, source_sha256_after=after, changed_source_files=changed))
        if self.status != "passed":
            raise RuntimeError("Simulator oracle did not close with verified source integrity")
