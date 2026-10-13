"""Pinned YAM grasp-site FK, NumPy only; no actuator or simulator dependency."""

import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

MODEL_SHA256 = "a7362a0d3e7bffd822b2ba9992c1fb4b00862eb86277a4f11326018680d9dcc3"


def rotation(axis, angle):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def transform(element):
    out = np.eye(4)
    out[:3, 3] = np.fromstring(element.get("pos", "0 0 0"), sep=" ")
    q = np.fromstring(element.get("quat", "1 0 0 0"), sep=" ")
    q /= np.linalg.norm(q)
    w, x, y, z = q
    out[:3, :3] = [[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                   [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                   [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]]
    return out


class YamKinematics:
    def __init__(self, model):
        raw = Path(model).read_bytes()
        if hashlib.sha256(raw).hexdigest() != MODEL_SHA256:
            raise ValueError("unsupported FK model; do not silently substitute geometry")
        root = ET.fromstring(raw)
        body = root.find("worldbody/body")
        self.chain = []
        for name in ("base", "link1", "link2", "link3", "link4", "link5", "link6", "gripper"):
            if body is None or body.get("name") != name:
                raise ValueError("unexpected YAM chain")
            joint = body.find("joint")
            axis = None
            if joint is not None:
                if joint.get("type") != "hinge" or joint.get("pos") != "0 0 0":
                    raise ValueError("unexpected hinge semantics")
                axis = np.fromstring(joint.get("axis"), sep=" ")
            self.chain.append((transform(body), axis))
            if name == "gripper":
                self.site = transform(body.find("site[@name='grasp_site']"))
            body = body.find("body")

    def position(self, joints):
        joints = np.asarray(joints, dtype=np.float64)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("FK requires six finite radian joints")
        out, i = np.eye(4), 0
        for fixed, axis in self.chain:
            out = out @ fixed
            if axis is not None:
                moving = np.eye(4)
                moving[:3, :3] = rotation(axis, joints[i])
                out = out @ moving
                i += 1
        return (out @ self.site)[:3, 3]

    def heights(self, state):
        return {arm: float(self.position(np.asarray(state)[offset:offset + 6])[2])
                for arm, offset in (("left", 0), ("right", 7))}

    def positions(self, joints):
        """Evaluate independent poses together, retaining the pinned float64 FK.

        This is geometry only: batching doesn't interpolate or alter any target.
        The scalar implementation remains the reference for offline validation.
        """
        joints = np.asarray(joints, dtype=np.float64)
        if joints.ndim != 2 or joints.shape[1] != 6 or not np.isfinite(joints).all():
            raise ValueError("batched FK requires finite (N,6) radian joints")
        out = np.broadcast_to(np.eye(4), (len(joints), 4, 4)).copy()
        i = 0
        for fixed, axis in self.chain:
            out = out @ fixed
            if axis is not None:
                x, y, z = axis / np.linalg.norm(axis)
                cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                angle = joints[:, i, None, None]
                moving = np.broadcast_to(np.eye(4), out.shape).copy()
                moving[:, :3, :3] = (np.eye(3) + np.sin(angle) * cross
                                     + (1 - np.cos(angle)) * (cross @ cross))
                out = out @ moving
                i += 1
        return (out @ self.site)[:, :3, 3]

    def gate_heights(self, state, base, contract):
        if not contract.get("grasp_z_range_m"):
            return self.heights(state)
        state, base = np.asarray(state), np.asarray(base)
        count = len(base)
        poses = np.vstack((state[:6], state[7:13], base[:, :6], base[:, 7:13]))
        z = self.positions(poses)[:, 2]
        return {"left": float(z[0]), "right": float(z[1]),
                "planned_z_m": {"left": z[2:2 + count].tolist(), "right": z[2 + count:].tolist()}}
