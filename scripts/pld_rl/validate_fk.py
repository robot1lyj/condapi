"""CPU geometry check against MuJoCo; no learner, model weights or robot I/O."""

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/pld-rl/src"))
from yam_pld_rl.kinematics import MODEL_SHA256
from yam_pld_rl.kinematics import YamKinematics


def main():
    import mujoco  # noqa: PLC0415 -- isolated server validation dependency

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    path = Path(__file__).resolve().parents[2] / "third_party/yam-fk-model/yam_linear4310.xml"
    fk = YamKinematics(path)
    xml = ET.parse(path).getroot()
    xml.remove(xml.find("asset"))
    for body in xml.iter("body"):
        for geom in body.findall("geom"):
            body.remove(geom)
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    indexes = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")] for i in range(1,7)]
    errors = []
    for joints in np.random.default_rng(179).uniform(-2,2,(1000,6)):
        data.qpos[indexes] = joints
        mujoco.mj_kinematics(model, data)
        errors.append(float(np.max(np.abs(fk.position(joints) - data.site_xpos[site]))))
    if max(errors) > 1e-10:
        raise AssertionError(f"NumPy FK mismatch: {max(errors)}m")
    result = {"status": "passed", "model_sha256": MODEL_SHA256, "mujoco": mujoco.__version__,
              "poses": len(errors), "max_coordinate_error_m": max(errors), "robot_calibration_verified": False,
              "scope": "same pinned MJCF kinematics only; no table calibration or limit verification"}
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
