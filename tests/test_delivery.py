import json
from pathlib import Path
import shutil
import tempfile
import unittest

from fastapi.testclient import TestClient

from fleetops.app import create_app
from fleetops.sensors import sample

ARTIFACTS = Path(__file__).resolve().parents[1]/"artifacts"

def payload(i, label=True):
    robot=65+i%10
    values,fault=sample(robot,i,i%4)
    return {"robot":f"sim-{robot}","sequence":i,"readings":values.tolist(),
            "fault_label":fault if label else None}

class DeliveryTest(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp=Path(tmp.name)
        self.db=self.tmp/"fleet.db"
        self.client=TestClient(create_app(ARTIFACTS,self.db))
        self.addCleanup(self.client.close)

    def test_simulation_split_metrics_and_export_parity(self):
        r=json.loads((ARTIFACTS/"report.json").read_text())
        self.assertTrue(r["simulated_data"])
        sets=[set(r["splits"][k]) for k in ("train_robot_ids","validation_robot_ids","test_robot_ids")]
        self.assertEqual(sum(map(len,sets)),len(set.union(*sets)))
        self.assertEqual(r["splits"]["test_windows"],780)
        for m in r["releases"].values():
            self.assertGreater(m["test"]["f1"],.85)
            self.assertLess(m["parity_max_abs_error"],1e-5)

    def test_dashboard_health_and_replay(self):
        self.assertEqual(self.client.get("/health").json()["active"],"v1")
        self.assertIn("Fleet Model Delivery",self.client.get("/").text)
        self.assertEqual(self.client.get("/static/dashboard.js").status_code,200)
        self.assertEqual(self.client.post("/api/replay?count=8").json()["windows"],8)
        self.assertEqual(self.client.get("/api/status").json()["total"],8)
        self.assertIn('fleet_windows_total{version="v1"} 8',self.client.get("/metrics").text)

    def test_invalid_and_duplicate_windows(self):
        row=payload(1)
        self.assertEqual(self.client.post("/api/infer",json={**row,"readings":row["readings"][:-1]}).status_code,422)
        self.assertEqual(self.client.post("/api/infer",json=row).status_code,200)
        self.assertEqual(self.client.post("/api/infer",json=row).status_code,409)
        self.assertEqual(self.client.get("/api/status").json()["total"],1)

    def test_offline_quality_gate_rejects_bad_candidate(self):
        target=self.tmp/"models"
        shutil.copytree(ARTIFACTS,target)
        r=json.loads((target/"report.json").read_text())
        r["releases"]["v2"]["test"]["f1"]=.55
        (target/"report.json").write_text(json.dumps(r))
        with TestClient(create_app(target,self.tmp/"other.db")) as client:
            res=client.post("/api/releases/canary",json={"candidate":"v2","percent":40})
            self.assertEqual(res.status_code,409)
            self.assertIsNone(client.get("/api/status").json()["state"]["candidate"])

    def test_checksum_rejects_corrupt_model(self):
        target=self.tmp/"models"
        shutil.copytree(ARTIFACTS,target)
        with (target/"v2.onnx").open("ab") as file:
            file.write(b"broken")
        with self.assertRaisesRegex(ValueError,"checksum mismatch"):
            create_app(target,self.tmp/"other.db")

    def test_canary_gates_promotion_and_rollback(self):
        self.assertEqual(self.client.post("/api/releases/canary",json={"candidate":"v2","percent":40}).status_code,200)
        self.assertEqual(self.client.post("/api/releases/promote").status_code,409)
        routes=set()
        for i in range(112):
            response=self.client.post("/api/infer",json=payload(i))
            self.assertEqual(response.status_code,200)
            prediction=response.json()
            routes.add(prediction["route"])
            if prediction["route"]=="canary":
                self.assertIsNotNone(prediction["baseline_probability"])
        self.assertEqual(routes,{"active","canary"})
        status=self.client.get("/api/status").json()
        self.assertGreaterEqual(status["trial"]["labeled"],16)
        self.assertGreaterEqual(status["trial"]["f1"],.85)
        self.assertEqual(self.client.post("/api/releases/promote").json()["active"],"v2")
        with TestClient(create_app(ARTIFACTS,self.db)) as reopened:
            self.assertEqual(reopened.get("/health").json()["active"],"v2")
            self.assertEqual(reopened.post("/api/releases/rollback").json()["active"],"v1")
            self.assertEqual(reopened.post("/api/infer",json=payload(200)).json()["model"],"v1")

    def test_unlabeled_windows_cannot_pass_promotion_gate(self):
        self.client.post("/api/releases/canary",json={"candidate":"v2","percent":50})
        for i in range(64):
            self.client.post("/api/infer",json=payload(i,label=False))
        self.assertGreater(self.client.get("/api/status").json()["trial"]["windows"],16)
        self.assertEqual(self.client.post("/api/releases/promote").status_code,409)
        self.assertIsNone(self.client.post("/api/releases/rollback").json()["candidate"])

if __name__=="__main__":
    unittest.main()
