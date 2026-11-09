"""FastAPI inference, release controls, and telemetry for a local fleet simulator."""
from contextlib import closing
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import sqlite3
from threading import RLock
from time import perf_counter, time_ns
from typing import Annotated, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, FiniteFloat

from .sensors import CHANNELS, STEPS, features, sample
from .train import metrics, session

SensorRow = Annotated[list[FiniteFloat], Field(min_length=5,max_length=5)]

class Window(BaseModel):
    robot: Annotated[str, Field(min_length=1,max_length=40,pattern=r"^[a-zA-Z0-9_-]+$")]
    sequence: Annotated[int, Field(ge=0,le=2**53-1)]
    readings: Annotated[list[SensorRow],Field(min_length=STEPS,max_length=STEPS)]
    fault_label: Literal[0,1] | None = None

class Start(BaseModel):
    candidate: Literal["v1","v2"] = "v2"
    percent: Annotated[int,Field(ge=5,le=50)] = 40

MIN_LABELED = 16
SCHEMA = """
CREATE TABLE IF NOT EXISTS release_state (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1), active TEXT NOT NULL, candidate TEXT,
  previous TEXT, percent INTEGER NOT NULL, trial INTEGER NOT NULL
);
INSERT OR IGNORE INTO release_state VALUES (1,'v1',NULL,NULL,0,0);
CREATE TABLE IF NOT EXISTS telemetry (
  id INTEGER PRIMARY KEY, observed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, robot TEXT NOT NULL,
  sequence INTEGER NOT NULL, model TEXT NOT NULL, trial INTEGER, probability REAL NOT NULL,
  alert INTEGER NOT NULL, label INTEGER, inference_ms REAL NOT NULL, baseline_probability REAL,
  UNIQUE(robot,sequence)
);
CREATE INDEX IF NOT EXISTS telemetry_trial ON telemetry(trial,id);
CREATE TABLE IF NOT EXISTS operations (
  id INTEGER PRIMARY KEY, observed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  event TEXT NOT NULL, detail TEXT NOT NULL
);
"""

def create_app(artifact_dir="artifacts", database="state/fleet.db"):
    artifacts = Path(artifact_dir)
    versions = json.loads((artifacts/"report.json").read_text())
    reports = versions["releases"]
    runtimes = {}
    for name in ("v1","v2"):
        meta = reports[name]
        path = artifacts/f"{name}.onnx"
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
            raise ValueError(f"Model checksum mismatch: {name}")
        runtimes[name] = session(path)
    database = Path(database)
    database.parent.mkdir(parents=True,exist_ok=True)
    lock = RLock()
    app = FastAPI(title="Fleet Model Delivery",version="1.0")
    static = Path(__file__).parent/"static"
    app.mount("/static",StaticFiles(directory=static),name="static")

    def connect():
        db=sqlite3.connect(database,timeout=10)
        db.row_factory=sqlite3.Row
        return db

    with closing(connect()) as db, db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript(SCHEMA)

    def state(db):
        data=dict(db.execute("SELECT * FROM release_state WHERE singleton=1").fetchone())
        del data["singleton"]
        return data

    def event(db,name,message):
        db.execute("INSERT INTO operations(event,detail) VALUES (?,?)",(name,message))

    def f1(name):
        return reports[name]["test"]["f1"]

    def offline(active,candidate):
        m=reports[candidate]
        # Use strict finite checks so malformed metrics cannot pass a release gate.
        checks=[
            ("Held-out F1",m["test"]["f1"],max(.85,f1(active)-.015),">="),
            ("Held-out recall",m["test"]["recall"],.85,">="),
            ("ONNX parity error",m["parity_max_abs_error"],1e-5,"<="),
            ("CPU p95 (ms)",m["cpu_p95_ms"],10,"<="),
        ]
        return [{"name":k,"value":v,"threshold":t,"operator":op,
                 "passed":bool(isfinite(v) and (v>=t if op==">=" else v<=t))}
                for k,v,t,op in checks]

    def trial_summary(db,s):
        rows=db.execute("SELECT label,alert,inference_ms FROM telemetry WHERE trial=?",
                        (s["trial"],)).fetchall() if s["candidate"] else []
        labeled=[r for r in rows if r["label"] is not None]
        positives=sum(r["label"] for r in labeled)
        return {"windows":len(rows),"labeled":len(labeled),"faults":positives,
                "f1":metrics([r["label"] for r in labeled],[r["alert"] for r in labeled])["f1"] if positives else None,
                "p95_ms":float(np.percentile([r["inference_ms"] for r in rows],95)) if rows else None}

    def probability(model,x):
        return float(runtimes[model].run(None,{"features":x})[0][0,0])

    def record(payload):
        x=features(np.asarray(payload.readings,dtype=np.float32))[None,:]
        if not np.isfinite(x).all():
            raise HTTPException(422,"Sensor values exceed usable numeric range")
        bucket=int.from_bytes(hashlib.sha256(f"{payload.robot}:{payload.sequence}".encode()).digest()[:4],"big")%100
        with lock, closing(connect()) as db, db:
            s=state(db)
            canary=bool(s["candidate"] and bucket<s["percent"])
            model=s["candidate"] if canary else s["active"]
            tick=perf_counter()
            prob=probability(model,x)
            elapsed=(perf_counter()-tick)*1000
            baseline=probability(s["active"],x) if canary else None
            alert=int(prob>=reports[model]["threshold"])
            try:
                db.execute("""INSERT INTO telemetry(robot,sequence,model,trial,probability,alert,label,inference_ms,baseline_probability)
                    VALUES (?,?,?,?,?,?,?,?,?)""",
                    (payload.robot,payload.sequence,model,s["trial"] if canary else None,prob,alert,payload.fault_label,elapsed,baseline))
            except sqlite3.IntegrityError as exc:
                raise HTTPException(409,"This robot window was already recorded") from exc
        return {"model":model,"route":"canary" if canary else "active","probability":prob,
                "alert":bool(alert),"inference_ms":elapsed,"baseline_probability":baseline}

    @app.get("/",include_in_schema=False)
    def dashboard():
        return FileResponse(static/"index.html")

    @app.get("/health")
    def health():
        with closing(connect()) as db:
            s=state(db)
        return {"status":"ready","active":s["active"],"versions":sorted(runtimes)}

    @app.post("/api/infer")
    def infer(payload: Window):
        return record(payload)

    @app.post("/api/replay")
    def replay(count: int = 96):
        if not 1<=count<=192:
            raise HTTPException(422,"Replay must contain 1 to 192 windows")
        run=time_ns()//1000
        result={"windows":count,"canary":0,"alerts":0}
        for i in range(count):
            robot=51+i%8
            x,y=sample(robot,run+i,i%4)
            r=record(Window(robot=f"sim-{robot}",sequence=run+i,readings=x.tolist(),fault_label=y))
            result["canary"]+=r["route"]=="canary"
            result["alerts"]+=r["alert"]
        return result

    @app.get("/api/status")
    def status():
        with lock, closing(connect()) as db:
            s=state(db)
            summary=trial_summary(db,s)
            total,alerts=db.execute("SELECT COUNT(*),COALESCE(SUM(alert),0) FROM telemetry").fetchone()
            recent=[dict(row) for row in db.execute("SELECT * FROM telemetry ORDER BY id DESC LIMIT 18")]
            history=[dict(row) for row in db.execute("SELECT * FROM operations ORDER BY id DESC LIMIT 8")]
        return {"state":s,"release_report":versions,"trial":summary,"total":total,"alerts":alerts,
                "recent":recent,"operations":history,"offline_gates":offline(s["active"],s["candidate"] or ("v2" if s["active"]=="v1" else "v1"))}

    @app.post("/api/releases/canary")
    def start_canary(request: Start):
        with lock, closing(connect()) as db, db:
            s=state(db)
            if s["candidate"] or request.candidate==s["active"]:
                raise HTTPException(409,"A distinct candidate and no active canary are required")
            checks=offline(s["active"],request.candidate)
            if not all(c["passed"] for c in checks):
                raise HTTPException(409,{"reason":"Candidate failed offline gates","gates":checks})
            db.execute("UPDATE release_state SET candidate=?,percent=?,trial=trial+1 WHERE singleton=1",
                       (request.candidate,request.percent))
            event(db,"Canary started",f"{request.candidate} receives {request.percent}% of hashed windows")
            return state(db)

    @app.post("/api/releases/promote")
    def promote():
        with lock, closing(connect()) as db, db:
            s=state(db)
            if not s["candidate"]:
                raise HTTPException(409,"Start a canary first")
            stats=trial_summary(db,s)
            if (stats["labeled"]<MIN_LABELED or stats["faults"]<2 or stats["f1"]<.85 or
                stats["p95_ms"]>30 or not all(g["passed"] for g in offline(s["active"],s["candidate"]))):
                raise HTTPException(409,{"reason":"Promotion requires at least 16 labeled canary windows, 2 faults, F1 >= 0.85, p95 <= 30 ms, and passing offline gates","trial":stats})
            db.execute("UPDATE release_state SET previous=active,active=candidate,candidate=NULL,percent=0 WHERE singleton=1")
            event(db,"Promoted",f'{s["candidate"]} replaced {s["active"]} after {stats["labeled"]} labeled canary windows')
            return state(db)

    @app.post("/api/releases/rollback")
    def rollback():
        with lock, closing(connect()) as db, db:
            s=state(db)
            if s["candidate"]:
                db.execute("UPDATE release_state SET candidate=NULL,percent=0 WHERE singleton=1")
                event(db,"Canary stopped",f'Returned all traffic to {s["active"]}')
            elif s["previous"]:
                db.execute("UPDATE release_state SET active=previous,previous=NULL WHERE singleton=1")
                event(db,"Rolled back",f'{s["previous"]} replaced {s["active"]}')
            else:
                raise HTTPException(409,"No canary or previous release to roll back")
            return state(db)

    @app.get("/metrics",response_class=PlainTextResponse)
    def prometheus():
        with closing(connect()) as db:
            rows=db.execute("SELECT model,COUNT(*),SUM(alert),SUM(inference_ms)/1000 FROM telemetry GROUP BY model").fetchall()
        return "\n".join(['# TYPE fleet_windows_total counter','# TYPE fleet_alerts_total counter',
            '# TYPE fleet_inference_seconds_total counter']+[line for v,n,a,t in rows for line in (
            f'fleet_windows_total{{version="{v}"}} {n}',f'fleet_alerts_total{{version="{v}"}} {a}',
            f'fleet_inference_seconds_total{{version="{v}"}} {t:.9f}')])+"\n"

    return app

def from_env():
    return create_app(os.getenv("FLEET_ARTIFACTS","artifacts"),os.getenv("FLEET_DATABASE","state/fleet.db"))
