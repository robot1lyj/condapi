"""Stage an immutable PLD bootstrap beside Pi, then upgrade with rollback."""
import argparse
import datetime
import json
from pathlib import Path

from deploy_pld_probe import await_normal_power
from deploy_pld_probe import inspect
from deploy_pld_probe import power_service
from deploy_pld_probe import ready
from deploy_pld_probe import run
from deploy_pld_probe import write


def create(args, source, name, port):
    command = source["Config"]["Cmd"].copy()
    if command[0].endswith("serve_pi05_rtc_trt.py"):
        command.pop(0)
    if "--pld-probe-only" in command:
        command.remove("--pld-probe-only")
    elif "--pld-bundle" in command:
        for flag in ("--pld-bundle", "--pld-encoder"):
            index = command.index(flag)
            del command[index:index+2]
    else:
        raise ValueError("this upgrade expects an existing PLD service")
    command[command.index("--port")+1] = str(port)
    command += ["--pld-bundle", "/pld-bundle", "--pld-encoder", "/pld-encoder.pth"]
    argv = ["docker", "create", "--name", name, "--restart", "no", "--runtime", "nvidia",
            "--network", "host", "--ipc", "host", "--security-opt", "label=disable"]
    for bind in source["HostConfig"]["Binds"]:
        if not bind.startswith("/home/wuyan-lyj/thor/pi/"):
            raise ValueError("unexpected source mount")
        if bind.split(":")[1] not in {"/deployment", "/pld-bundle", "/pld-encoder.pth", "/pld-cases"}:
            argv += ["-v", bind]
    argv += ["-v", f"{args.release}:/deployment", "-v", f"{args.bundle}:/pld-bundle:ro",
             "-v", f"{args.encoder}:/pld-encoder.pth:ro", "-v", f"{args.cases}:/pld-cases:ro"]
    allowed = {"OPENPI_DATA_HOME", "JAX_PLATFORMS", "HF_HUB_OFFLINE", "WANDB_MODE",
               "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"}
    for value in source["Config"]["Env"]:
        if value.split("=", 1)[0] in allowed:
            argv += ["-e", value]
    argv += ["-e", f"PLD_HEALTH_PORT={port}", args.image, *command]
    run(argv)
    write(args.release/f"{name}-created.json", {"argv": argv, "image": inspect(name)["Image"]})


def smoke(args, name, port, tag):
    ready(name, f"http://192.168.250.1:{port}")
    filename = f"{tag}-bundle-smoke.json"
    import hashlib  # noqa: PLC0415
    manifest = json.loads((args.bundle/"manifest.json").read_text())
    run(["docker", "exec", name, "python", "/opt/pld-rl/scripts/thor/smoke_pld_bundle_ws.py",
         "--url", f"ws://192.168.250.1:{port}", "--cases", "/pld-cases/cases.json",
         "--expected-actor-phase", manifest["phase"],
         "--bundle-manifest", "/pld-bundle/manifest.json",
         "--expected-snapshot-sha256", hashlib.sha256((args.bundle/"manifest.json").read_bytes()).hexdigest(),
         "--output", f"/deployment/{filename}"], capture=False)
    result = json.loads((args.release/filename).read_text())
    if result["status"] != "passed":
        raise ValueError("recorded-input or latency gate failed")
    run(["docker", "exec", name, "python", "/opt/pld-rl/scripts/thor/smoke_pi05_rtc_ws.py",
         "--url", f"ws://192.168.250.1:{port}", "--cases", "/cases/cases.json",
         "--expected-weights-sha256", result["metadata"]["checkpoint_weights_sha256"],
         "--expected-prefix-norm", "quantile", "--output", f"/deployment/{tag}-rtc-smoke.json"], capture=False)
    return filename, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("preflight", "cutover"), required=True)
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--maxn-helper", type=Path, required=True)
    p.add_argument("--watch-helper", type=Path, required=True)
    args = p.parse_args()
    source = inspect("pld-rl")
    image_id = json.loads(run(["docker", "image", "inspect", args.image]))[0]["Id"]
    candidate = "pld-rl-preflight-" + args.release.name
    if args.phase == "preflight":
        write(args.release/"source-container.json", {"id": source["Id"], "image": source["Image"],
              "command": source["Config"]["Cmd"], "mounts": source["HostConfig"]["Binds"]})
        create(args, source, candidate, 8001)
        if "NV Power Mode: MAXN" in run(["nvpmodel", "-q"]):
            run(["docker", "start", candidate])
        else:
            power_service(args, candidate, "thor-"+args.release.name+"-preflight-maxn")
        try:
            filename, result = smoke(args, candidate, 8001, "preflight")
        finally:
            run(["docker", "stop", candidate])
        write(args.release/"preflight.json", {"status": "passed", "source_id": source["Id"],
              "image_id": image_id, "snapshot_sha256": result["metadata"]["pld"]["snapshot_sha256"],
              "report": filename})
        return
    receipt = json.loads((args.release/"preflight.json").read_text())
    if receipt["status"] != "passed" or receipt["source_id"] != source["Id"] or receipt["image_id"] != image_id:
        raise ValueError("preflight identity changed")
    retained = "pld-rl-retained-" + args.release.name
    run(["docker", "stop", "pld-rl"])
    run(["docker", "rename", "pld-rl", retained])
    try:
        await_normal_power()
        create(args, source, "pld-rl", 8000)
        power_service(args, "pld-rl", "thor-" + args.release.name + "-maxn")
        filename, result = smoke(args, "pld-rl", 8000, "production")
        if result["metadata"]["pld"]["snapshot_sha256"] != receipt["snapshot_sha256"]:
            raise ValueError("production actor differs from preflight")
        power = run(["nvpmodel", "-q"])
        if "NV Power Mode: MAXN" not in power:
            raise ValueError("production MAXN is not held")
        manifest = json.loads((args.bundle/"manifest.json").read_text())
        learned = manifest["phase"] == "candidate_requires_AB"
        write(args.release/"deployment.json", {"status": "online_learned_candidate_requires_AB" if learned else "online_bootstrap_collect_ready",
              "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),  # noqa: UP017 -- host Python 3.10 compatible
              "image_id": image_id, "container_id": inspect("pld-rl")["Id"], "metadata": result["metadata"],
              "report": filename, "power": power, "retained_container": retained,
              "robot_rollout_of_new_actor": False, "cross_IPC_new_actor_tested": False,
              "training_updates": manifest["provenance"].get("actor_updates", 0)})
    except Exception:
        run(["docker", "stop", "pld-rl"])
        run(["docker", "rename", "pld-rl", "pld-rl-failed-" + args.release.name])
        await_normal_power()
        run(["docker", "rename", retained, "pld-rl"])
        power_service(args, "pld-rl", "thor-" + args.release.name + "-rollback")
        raise


if __name__ == "__main__":
    main()
