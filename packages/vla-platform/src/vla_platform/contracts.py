"""Versioned wire contracts, independent of NumPy, Torch and JAX."""

import math


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_contract(contract):
    require(contract.get("schema_version") == 1, "Unsupported robot contract version")
    require(isinstance(contract.get("id"), str) and contract["id"], "Missing contract id")
    order = contract.get("action_order", [])
    require(isinstance(order, list) and len(order) > 0, "Missing action order")
    require(all(isinstance(x, str) and x for x in order), "Invalid action names")
    require(len(set(order)) == len(order), "Duplicate action names")
    require(contract.get("state_dim") == len(order), "State/action dimensions must agree")
    cameras = contract.get("cameras", [])
    require(isinstance(cameras, list) and bool(cameras), "Missing cameras")
    require(all(isinstance(x, str) and x for x in cameras), "Invalid camera names")
    require(len(set(cameras)) == len(cameras), "Duplicate cameras")
    require(contract.get("output_semantics") == "absolute", "Declare absolute robot-space output")
    require(contract.get("units_status") in ("unverified", "audited"), "Missing unit audit status")
    return contract


def validate_request(request, contract):
    validate_contract(contract)
    require(request.get("schema_version") == 1, "Unsupported request schema")
    require(request.get("contract_id") == contract["id"], "Request contract mismatch")
    for field in ("request_id", "session_id", "prompt"):
        require(isinstance(request.get(field), str) and bool(request[field]), f"Missing {field}")
    require(number(request.get("timestamp_s")) and request["timestamp_s"] >= 0, "Invalid timestamp")
    state = request.get("state", [])
    require(isinstance(state, list) and len(state) == contract["state_dim"], "State dimension mismatch")
    require(all(number(x) for x in state), "State must be finite")
    images = request.get("images", {})
    require(isinstance(images, dict) and set(images) == set(contract["cameras"]), "Camera set mismatch")
    require(all(isinstance(x, str) and bool(x) for x in images.values()), "Expected image file references")
    require(type(request.get("reset", False)) is bool, "reset must be boolean")
    return request


def validate_response(response, contract):
    validate_contract(contract)
    require(response.get("schema_version") == 1, "Unsupported response schema")
    require(response.get("contract_id") == contract["id"], "Response contract mismatch")
    for field in ("request_id", "session_id", "model_version"):
        require(isinstance(response.get(field), str) and bool(response[field]), f"Missing {field}")
    actions = response.get("actions", [])
    require(isinstance(actions, list) and bool(actions), "Empty action chunk")
    for row in actions:
        require(isinstance(row, list) and len(row) == contract["state_dim"], "Action dimension mismatch")
        require(all(number(x) for x in row), "Actions must be finite")
    require(number(response.get("action_dt_s")) and response["action_dt_s"] > 0, "Invalid action period")
    require(number(response.get("latency_ms")) and response["latency_ms"] >= 0, "Invalid latency")
    require(response.get("output_semantics") == contract["output_semantics"], "Action semantics mismatch")
    return response
