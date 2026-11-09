"""Send simulated telemetry to a running HTTP service."""
import argparse
import json
from time import time_ns
from urllib.request import Request, urlopen
from .sensors import sample

if __name__ == "__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--url",default="http://127.0.0.1:8000")
    cli.add_argument("--count",type=int,default=96)
    args=cli.parse_args()
    run=time_ns()//1000
    routes={"active":0,"canary":0}
    for i in range(args.count):
        robot=51+i%8
        x,y=sample(robot,run+i,i%4)
        body=json.dumps({"robot":f"sim-{robot}","sequence":run+i,"readings":x.tolist(),"fault_label":y})
        with urlopen(Request(args.url.rstrip("/")+"/api/infer",body.encode(),
                             {"Content-Type":"application/json"}),timeout=10) as response:
            routes[json.load(response)["route"]]+=1
    print(json.dumps(routes,indent=2))
