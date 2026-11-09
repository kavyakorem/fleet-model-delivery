"""Train, export, and evaluate both releases before the service starts."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

from .sensors import WIDTH, dataset

TRAIN, VALID, TEST = range(1, 31), range(31, 39), range(39, 51)

def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x, -35, 35)))

def predict(x, params):
    center, scale, w, b, v, c = params
    return sigmoid(np.maximum(((x-center)/scale)@w+b, 0)@v+c).ravel()

def fit(x, y, seed, hidden, epochs):
    rng = np.random.default_rng(seed)
    center, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-5)
    z = (x-center)/scale
    w = rng.normal(0, .18, (WIDTH, hidden)).astype(np.float32)
    b = np.zeros(hidden, np.float32)
    v = rng.normal(0, .12, (hidden, 1)).astype(np.float32)
    c = np.zeros(1, np.float32)
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(len(z)), max(1, len(z)//96)):
            inp = z[idx]
            h = inp@w+b
            relu = np.maximum(h, 0)
            err = (sigmoid(relu@v+c)-y[idx,None]).astype(np.float32)/len(idx)
            upstream = (err@v.T)*(h>0)
            w -= .075*(inp.T@upstream+.001*w)
            b -= .075*upstream.sum(0)
            v -= .075*(relu.T@err+.001*v)
            c -= .075*err.sum(0)
    return center, scale, w, b, v, c

def metrics(y, predicted):
    y, predicted = np.asarray(y, dtype=bool), np.asarray(predicted, dtype=bool)
    tp, fp, fn = int((y&predicted).sum()), int((~y&predicted).sum()), int((y&~predicted).sum())
    return {"precision": tp/max(1,tp+fp), "recall": tp/max(1,tp+fn),
            "f1": 2*tp/max(1,2*tp+fp+fn), "accuracy": float((y==predicted).mean())}

def session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])

def export(path, params):
    nodes = [
        helper.make_node(op, inputs, [output]) for op, inputs, output in [
            ("Sub", ["features","center"], "centered"),
            ("Div", ["centered","scale"], "normalized"),
            ("Gemm", ["normalized","w","b"], "hidden"),
            ("Relu", ["hidden"], "active"),
            ("Gemm", ["active","v","c"], "logit"),
            ("Sigmoid", ["logit"], "probability")]]
    graph = helper.make_graph(nodes, "fault-risk", [
        helper.make_tensor_value_info("features",TensorProto.FLOAT,["batch",WIDTH])], [
        helper.make_tensor_value_info("probability",TensorProto.FLOAT,["batch",1])],
        [numpy_helper.from_array(np.asarray(arr,dtype=np.float32),name)
         for name,arr in zip(("center","scale","w","b","v","c"),params)])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",17)], ir_version=9)
    onnx.checker.check_model(model)
    onnx.save(model,path)

def train(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    x,y = dataset(TRAIN)
    xv,yv = dataset(VALID)
    xt,yt = dataset(TEST)
    releases = {}
    for version, hidden, epochs, count in [("v1",8,45,975),("v2",20,100,len(x))]:
        params = fit(x[:count],y[:count],42,hidden,epochs)
        val = predict(xv,params)
        threshold = max(np.linspace(.25,.75,21),key=lambda t: metrics(yv,val>=t)["f1"])
        file = directory/f"{version}.onnx"
        export(file,params)
        runtime = session(file)
        output = runtime.run(None,{"features":xt})[0].ravel()
        latency = []
        for row in xt[:160]:
            start = perf_counter()
            runtime.run(None,{"features":row[None,:]})
            latency.append((perf_counter()-start)*1000)
        releases[version] = {"file":file.name,"sha256":hashlib.sha256(file.read_bytes()).hexdigest(),
            "threshold":float(threshold),"test":metrics(yt,output>=threshold),
            "parity_max_abs_error":float(np.max(np.abs(predict(xt,params)-output))),
            "cpu_p95_ms":float(np.percentile(latency[10:],95)),
            "training_windows":count}
    report = {"simulated_data":True,"splits":{"train_robot_ids":list(TRAIN),
        "validation_robot_ids":list(VALID),"test_robot_ids":list(TEST),"test_windows":len(xt)},
        "releases":releases}
    (directory/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    return report

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="artifacts")
    args=parser.parse_args()
    report=train(args.output)
    print(json.dumps(report["releases"],indent=2))
