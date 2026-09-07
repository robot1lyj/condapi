"""Single-client TensorRT policy for the audited fixed three-view H50 engine.

No PyTorch weight loading, CPU fallback, dtype conversion or hidden reshaping.
Use the existing Policy transforms so measured calls still include preprocessing
and YAM output conversion. Production concurrency/IPC is a separate integration.
"""

import json
from pathlib import Path

from benchmark_pi05 import digest
import numpy as np
from onnx_sampler import INPUT_NAMES
from onnx_sampler import check_text_bucket
from onnx_sampler import flat_inputs
import torch

from openpi import transforms
from openpi.policies.policy import Policy


def compare_export_reference(path, normalized):
    """Measure engine-only drift against the exact eager export preparation."""
    with np.load(path, allow_pickle=False) as arrays:
        reference = arrays["reference"]
        if reference.shape != (50, 32) or not np.array_equal(reference, arrays["prepared"]):
            raise ValueError("Export reference is not an equivalent H50 preparation")
    if normalized.ndim != 3 or normalized.shape[1:] != (50, 32):
        raise ValueError("Expected repeated H50 engine outputs")
    if not np.isfinite(reference).all() or not np.isfinite(normalized).all():
        raise ValueError("Non-finite export reference or engine outputs")
    difference = np.abs(normalized.astype(np.float64) - reference.astype(np.float64))
    return {
        "reference_sha256": digest(path),
        "normalized_32d_max_abs": float(difference.max()),
        "normalized_14d_max_abs": float(difference[..., :14].max()),
        "normalized_14d_mae": float(difference[..., :14].mean()),
        "exact": bool(np.all(difference == 0)),
        "scope": "engine vs same BF16 eager export preparation; not JAX or task acceptance",
    }


class TensorRTModel(torch.nn.Module):
    def __init__(self, engine_dir):
        super().__init__()
        import tensorrt as trt  # noqa: PLC0415

        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT CUDA execution required")
        self.device = torch.device("cuda", torch.cuda.current_device())
        engine_dir = Path(engine_dir)
        self.build_report = json.loads((engine_dir / "engine_report.json").read_text())
        report = self.build_report
        if report["status"] != "built_not_inference_or_accuracy_validated" or report["exit_code"] != 0:
            raise ValueError("A successful engine build is required")
        if report["tf32"] or report["quantization"] is not None or not report["strongly_typed"]:
            raise ValueError("This candidate requires non-quantized, strongly typed, TF32-off engine")
        source_report = Path(report["source_export"]) / "export_report.json"
        if digest(source_report) != report["source_export_report_sha256"]:
            raise ValueError("Source export report fingerprint mismatch")
        export = json.loads(source_report.read_text())
        self.text_bucket = export.get("text_bucket", 200)
        if not 1 <= self.text_bucket <= 200:
            raise ValueError("Unsupported text bucket")
        if self.text_bucket < 200 and export.get("padding_experiment", {}).get("status") != (
            "offline_experiment_supported_not_accuracy_approved"
        ):
            raise ValueError("Padding engine lacks its separately recorded diagnostic evidence")
        engine_file = engine_dir / "sampler.engine"
        if digest(engine_file) != report["engine_sha256"]:
            raise ValueError("Engine hash mismatch")
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_file.read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT deserialization failed")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT context creation failed")
        types = {trt.float32: torch.float32, trt.int64: torch.int64, trt.int32: torch.int32, trt.bool: torch.bool}
        self.inputs, self.outputs, self.io_contract = {}, {}, {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = types.get(self.engine.get_tensor_dtype(name))
            if dtype is None or any(dim <= 0 for dim in shape):
                raise ValueError(f"Unexpected dynamic shape or IO dtype: {name}")
            if self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                raise ValueError(f"Expected CUDA data tensor: {name}")
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            self.io_contract[name] = {"shape": list(shape), "dtype": str(dtype), "input": is_input}
            if is_input:
                self.inputs[name] = (shape, dtype)
            else:
                self.outputs[name] = torch.empty(shape, dtype=dtype, device=self.device)
                self.context.set_tensor_address(name, self.outputs[name].data_ptr())
        required = {"images", "img_masks", "lang_tokens", "lang_masks", "noise"}
        if not required.issubset(self.inputs) or not set(self.inputs).issubset(INPUT_NAMES):
            raise ValueError("Engine dropped required model inputs or added unknown inputs")
        if (
            set(self.outputs) != {"actions"}
            or tuple(self.outputs["actions"].shape) != (1, 50, 32)
            or self.outputs["actions"].dtype != torch.float32
        ):
            raise ValueError("Engine output must be exactly (1,50,32)")
        self.unused_bindings = {}
        self.graph_inputs = None
        self.cuda_graph = None
        # Pi0.5 encodes state in prompt tokens and retains the original state
        # for output transforms. Its separate sampler state input is dead.
        # TRT retains this unused ONNX DOUBLE input as FLOAT; bind a dummy
        # only after proving it has no graph consumers. Never cast real state.
        if "state" in self.inputs:
            import onnx  # noqa: PLC0415

            source = Path(report["source_export"])
            onnx_path = source / "sampler.onnx"
            if digest(onnx_path) != report["source_onnx_sha256"]:
                raise ValueError("Source ONNX fingerprint mismatch")
            graph = onnx.load(str(onnx_path), load_external_data=False).graph
            used = {name for node in graph.node for name in node.input} | {item.name for item in graph.output}
            if "state" not in used:
                shape, dtype = self.inputs["state"]
                self.unused_bindings["state"] = torch.zeros(shape, dtype=dtype, device=self.device)
                self.io_contract["state"]["unused_binding"] = "zero dummy; original state untouched in policy"

    def enable_cuda_graph(self):
        self.graph_inputs = {
            name: torch.empty(shape, dtype=dtype, device=self.device) for name, (shape, dtype) in self.inputs.items()
        }

    def _execute(self):
        if not self.context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execution failed")

    @torch.no_grad()
    def validate_graph_current(self):
        if self.cuda_graph is None:
            raise RuntimeError("No TensorRT graph captured")
        self.cuda_graph.replay()
        captured = self.outputs["actions"].clone()
        self._execute()
        reference = self.outputs["actions"]
        if not torch.isfinite(reference).all() or not torch.equal(captured, reference):
            raise RuntimeError("TensorRT CUDA graph changed the uncaptured engine output")
        return 0.0

    @torch.no_grad()
    def sample_actions(self, device, observation, *, noise=None, num_steps=10):
        if noise is None or num_steps != 10:
            raise ValueError("This engine requires ten denoising steps and explicit noise")
        if self.text_bucket < 200:
            check_text_bucket(observation.tokenized_prompt, observation.tokenized_prompt_mask, self.text_bucket)
        values = dict(zip(INPUT_NAMES, flat_inputs(observation, noise), strict=True))
        keepalive = []
        for name, (shape, dtype) in self.inputs.items():
            value = values[name]
            if name in self.unused_bindings:
                if tuple(value.shape) != shape or value.device != self.device:
                    raise ValueError(f"Unused input shape/device mismatch: {name}")
                value = self.unused_bindings[name]
            if tuple(value.shape) != shape or value.dtype != dtype or value.device != self.device:
                raise ValueError(f"Input contract mismatch: {name}")
            value = value.contiguous()
            if self.graph_inputs is not None:
                self.graph_inputs[name].copy_(value)
                value = self.graph_inputs[name]
            keepalive.append(value)
            if not self.context.set_tensor_address(name, value.data_ptr()):
                raise RuntimeError(f"Failed to bind TensorRT input: {name}")
        if self.graph_inputs is None:
            self._execute()
        else:
            if self.cuda_graph is None:
                self._execute()  # Flush TensorRT's lazy initialization before capture.
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self._execute()
                self.cuda_graph = graph
            self.cuda_graph.replay()
        return self.outputs["actions"].clone()


def create_trt_policy(train_config, engine_dir, norm_stats):
    # Same ordered transforms as create_trained_policy with no repack/default
    # prompt overrides. The recorded suite already supplies its real prompt.
    data = train_config.data.create(train_config.assets_dirs, train_config.model)
    return Policy(
        TensorRTModel(engine_dir),
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data.use_quantile_norm),
            *data.model_transforms.inputs,
        ],
        output_transforms=[
            *data.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data.use_quantile_norm),
            *data.data_transforms.outputs,
        ],
        sample_kwargs={"num_steps": 10},
        metadata=train_config.policy_metadata,
        is_pytorch=True,
        pytorch_device="cuda",
    )
