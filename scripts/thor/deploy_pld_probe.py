"""Thor-only staged PLD probe deployment using an existing validated Pi service.

Preflight runs beside the old service on 8001. Cutover requires those real
inference receipts, preserves the stopped old container, transfers MAXN to a
new transient service, and restores the old service on failed validation.
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def run(argv, *, capture=True):
    return subprocess.check_output(argv, text=True).strip() if capture else subprocess.check_call(argv)


def inspect(name):
    return json.loads(run(["docker","inspect",name]))[0]


def write(path, value):
    with path.open("x") as f:
        json.dump(value,f,indent=2)
        f.write("\n")


def make_container(args, original, name, port):
    command = original["Config"]["Cmd"]
    if not command[0].endswith("serve_pi05_rtc_trt.py") or any(x.startswith("--pld") for x in command):
        raise ValueError("source is not the original trained-RTC service")
    command = command[1:].copy()
    command[command.index("--port")+1] = str(port)
    command.append("--pld-probe-only")
    argv = ["docker","create","--name",name,"--restart","no","--runtime","nvidia","--network","host","--ipc","host",
            "--security-opt","label=disable","--label","ai.wuyan.role=PLD-Grasp-probe"]
    for bind in original["HostConfig"]["Binds"]:
        if not bind.startswith("/home/wuyan-lyj/thor/pi/"):
            raise ValueError("source mounts outside the Thor Pi workspace")
        argv += ["-v",bind]
    argv += ["-v",f"{args.release}:/deployment"]
    allowed = {"OPENPI_DATA_HOME","JAX_PLATFORMS","HF_HUB_OFFLINE","WANDB_MODE","TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"}
    for value in original["Config"]["Env"]:
        if value.split("=",1)[0] in allowed:
            argv += ["-e",value]
    argv += ["-e",f"PLD_HEALTH_PORT={port}",args.image,*command]
    identifier = run(argv)
    write(args.release/f"{name}-created.json",{"container_id":identifier,"argv":argv,"image_id":inspect(name)["Image"]})


def ready(name, url, timeout=120):
    start = time.monotonic()
    while time.monotonic()-start < timeout:
        state = inspect(name)["State"]
        if not state["Running"]:
            if state["Status"] == "created":
                time.sleep(1)
                continue
            raise RuntimeError(f"{name} exited; inspect its retained logs")
        check = subprocess.run(["docker","exec",name,"python","-c",
                                "import urllib.request; assert urllib.request.urlopen("+repr(url+"/healthz")+",timeout=2).read().strip()==b'OK'"],
                               stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
        if check.returncode == 0:
            return
        time.sleep(1)
    raise TimeoutError("inference service readiness timed out")


def smoke(args, name, port, weights, tag):
    url = f"http://192.168.250.1:{port}"
    ready(name,url)
    reports = []
    for script in ("smoke_pi05_rtc_ws.py","smoke_pld_probe_ws.py"):
        output = f"{tag}-{script.removesuffix('.py')}.json"
        command = ["docker","exec",name,"python",f"/opt/pld-rl/scripts/thor/{script}","--url",url.replace("http:","ws:"),
                   "--cases","/cases/cases.json","--expected-weights-sha256",weights,"--output",f"/deployment/{output}"]
        if script == "smoke_pi05_rtc_ws.py":
            command += ["--expected-prefix-norm","quantile"]
        run(command,capture=False)
        report = json.loads((args.release/output).read_text())
        if report["status"] != "passed":
            raise ValueError("real inference smoke failed")
        reports.append(output)
    return reports


def power_service(args, container, unit):
    # Docker is already the authorized root deployment interface. This small
    # ephemeral client only connects to the host systemd manager; production
    # model containers get neither systemd sockets nor extra capabilities.
    argv = ["docker","run","--rm","--network","none","--pid","host","--user","0","--cap-drop","ALL",
            "--security-opt","no-new-privileges",
            "-v","/usr/bin/systemd-run:/systemd-run:ro",
            "-v","/usr/lib/aarch64-linux-gnu/systemd:/usr/lib/aarch64-linux-gnu/systemd:ro",
            "-v","/run/systemd:/run/systemd:ro","--entrypoint","/systemd-run",args.image,
            f"--unit={unit}","--collect","--service-type=exec","/usr/bin/python3",str(args.maxn_helper),"--",
            "/bin/bash","-c",f"docker start {shlex.quote(container)} && exec python3 {shlex.quote(str(args.watch_helper))} --container {shlex.quote(container)}"]
    run(argv,capture=False)


def await_normal_power(timeout=40):
    start = time.monotonic()
    while time.monotonic()-start < timeout:
        if "NV Power Mode: 120W" in run(["nvpmodel","-q"]):
            time.sleep(1)  # allow the previous wrapper to release its lock
            return
        time.sleep(1)
    raise TimeoutError("previous MAXN session did not restore 120W")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=["preflight","cutover"],required=True)
    p.add_argument("--release",type=Path,required=True)
    p.add_argument("--image",required=True)
    p.add_argument("--source-container",default="pi05-rtc-infer")
    p.add_argument("--preflight-container",default="pld-rl-preflight-20261012-r1")
    p.add_argument("--maxn-helper",type=Path,required=True)
    p.add_argument("--watch-helper",type=Path,required=True)
    args = p.parse_args()
    args.release = args.release.resolve()
    original = inspect(args.source_container)
    image = json.loads(run(["docker","image","inspect",args.image]))[0]
    if args.phase == "preflight":
        if not original["State"]["Running"] or "NV Power Mode: MAXN" not in run(["nvpmodel","-q"]):
            raise RuntimeError("preflight requires the existing active MAXN session")
        write(args.release/"source-container.json",{"name":args.source_container,"id":original["Id"],"image":original["Image"],
              "command":original["Config"]["Cmd"],"mounts":original["Mounts"],"state":original["State"]["Status"]})
        make_container(args,original,args.preflight_container,8001)
        run(["docker","start",args.preflight_container])
        weights = run(["docker","exec",args.preflight_container,"python","-c",
                       "import json; print(json.load(open('/ckpt/rtc_manifest.json'))['model_weights_sha256'])"])
        reports = smoke(args,args.preflight_container,8001,weights,"preflight")
        run(["docker","stop",args.preflight_container])
        write(args.release/"preflight.json",{"status":"passed","image_id":image["Id"],"weights_sha256":weights,
              "source_container_id":original["Id"],"reports":reports,"container":args.preflight_container})
        print("PREFLIGHT_PASSED; original Pi remains online",flush=True)
        return
    receipt = json.loads((args.release/"preflight.json").read_text())
    if receipt["status"] != "passed" or receipt["image_id"] != image["Id"] or receipt["source_container_id"] != original["Id"]:
        raise ValueError("preflight identity changed")
    make_container(args,original,"pld-rl",8000)
    print("Stopping original Pi and transferring MAXN to pld-rl",flush=True)
    run(["docker","stop",args.source_container])
    try:
        await_normal_power()
        power_service(args,"pld-rl","thor-pld-rl-maxn-20261012-r1")
        time.sleep(1)
        reports = smoke(args,"pld-rl",8000,receipt["weights_sha256"],"production")
        power = run(["nvpmodel","-q"])
        if "NV Power Mode: MAXN" not in power:
            raise RuntimeError("new inference service is not held in MAXN")
        write(args.release/"deployment.json",{"status":"online_probe_only","observed_at":datetime.datetime.now(datetime.UTC).isoformat(),
              "container":"pld-rl","container_id":inspect("pld-rl")["Id"],"image_id":image["Id"],
              "source_revision":image["Config"]["Labels"]["org.opencontainers.image.revision"],
              "deploy_controller_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "url":"ws://192.168.250.1:8000","weights_sha256":receipt["weights_sha256"],"reports":reports,
              "previous_container_state":inspect(args.source_container)["State"]["Status"],"power":power,
              "actor_loaded":False,"collect_enabled":False,"cross_IPC_tested":False})
        print("ONLINE pld-rl; original container stopped and retained",flush=True)
    except Exception:
        run(["docker","stop","pld-rl"])
        await_normal_power()
        power_service(args,args.source_container,"thor-pi-rollback-pld-20261012-r1")
        print("Restoring original Pi; failed candidate retained",flush=True)
        raise


if __name__ == "__main__":
    main()
