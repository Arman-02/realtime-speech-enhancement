from __future__ import annotations

import json

import numpy as np
import onnx
import pytest
import torch

from ans.dsp import HOP, STFT
from ans.export_onnx import chirp_signal, export
from ans.models import build_model


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    manifest = export(None, out, int8=True, int8_static=True, calib_clips=4, seed=0)
    return out, manifest


def test_manifest_schema(exported):
    out, manifest = exported
    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk == manifest
    for key in [
        "name",
        "version",
        "trained",
        "sample_rate",
        "n_fft",
        "hop",
        "window",
        "onnx",
        "inputs",
        "outputs",
        "state_shape",
        "params",
        "macs_per_frame",
        "metrics",
        "variant",
        "lookahead_frames",
        "atten_lim_db",
        "budget",
        "training",
    ]:
        assert key in manifest
    assert manifest["trained"] is False and manifest["version"] == "0.2.0"
    assert manifest["inputs"] == {"spec": [1, 1, 161, 2], "state_in": [2, 1, 256], "atten_lim_db": [1]}
    assert manifest["atten_lim_db"] == {"default": 100, "min": 0, "max": 100}
    assert manifest["lookahead_frames"] == 0
    b = manifest["budget"]
    for k in ("weights_kb_fp32", "weights_kb_int8", "peak_activation_kb", "state_kb", "mmacs_per_s"):
        assert b[k] > 0
    assert b["weights_kb_int8"] < b["weights_kb_fp32"]
    assert manifest["outputs"] == {"spec_out": [1, 1, 161, 2], "state_out": [2, 1, 256]}
    assert manifest["window"] == "sqrt_hann_periodic"


def test_onnx_graph_static_opset17(exported):
    out, _ = exported
    proto = onnx.load(str(out / "ans_stream.onnx"))
    assert any(o.domain in ("", "ai.onnx") and o.version == 17 for o in proto.opset_import)
    assert not {"If", "Loop", "Scan"} & {n.op_type for n in proto.graph.node}
    for v in list(proto.graph.input) + list(proto.graph.output):
        assert all(d.dim_value > 0 for d in v.type.tensor_type.shape.dim), v.name


def test_ort_matches_torch(exported):
    import onnxruntime as ort

    out, _ = exported
    sess = ort.InferenceSession(str(out / "ans_stream.onnx"), providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["spec", "state_in", "atten_lim_db"]
    torch.manual_seed(0)
    model = build_model("ans_crn").eval()
    rng = np.random.default_rng(5)
    st_t, st_o = model.init_state(1), np.zeros((2, 1, 256), np.float32)
    for i in range(30):
        spec = rng.standard_normal((1, 1, 161, 2)).astype(np.float32)
        lim = np.array([100.0 if i % 2 else 9.0], np.float32)
        with torch.no_grad():
            o_t, st_t = model(torch.from_numpy(spec), st_t, atten_lim_db=torch.from_numpy(lim))
        o_o, st_o = sess.run(None, {"spec": spec, "state_in": st_o, "atten_lim_db": lim})
        assert np.abs(o_t.numpy() - o_o).max() < 1e-4
        assert np.abs(st_t.numpy() - st_o).max() < 1e-4


def test_test_vectors_consistent(exported):
    out, _ = exported
    tv = json.loads((out / "test_vectors.json").read_text())
    x = np.asarray(tv["input_wav"], np.float32)
    np.testing.assert_allclose(x, chirp_signal(), atol=1e-6)
    frames = np.asarray(tv["stft_frames"], np.float32)
    assert frames.shape == (8, 161, 2)
    ref = STFT()(torch.from_numpy(x), center=False)[:8].numpy()
    np.testing.assert_allclose(frames, ref, atol=1e-4)
    assert np.asarray(tv["model_out_frames"]).shape == (8, 161, 2)
    # full streaming output == offline torch enhancement delayed by one hop
    torch.manual_seed(0)
    model = build_model("ans_crn").eval()
    with torch.no_grad():
        offline = model.enhance(torch.from_numpy(x)[None])[0].numpy()
    y = np.asarray(tv["output_wav"], np.float32)
    assert len(y) == len(x)
    np.testing.assert_allclose(y[HOP:], offline[:-HOP], atol=1e-3)
    # v0.2: explicit main-case limit and the atten_lim_db = 12 case (web format: top-level "atten_case")
    assert tv["atten_lim_db"] == 100
    case = tv["atten_case"]
    assert case["atten_lim_db"] == 12
    assert np.asarray(case["model_out_frames"]).shape == (8, 161, 2)
    with torch.no_grad():
        off12 = model.enhance(torch.from_numpy(x)[None], atten_lim_db=12.0)[0].numpy()
        frames_t = torch.from_numpy(frames)[None]
        m12, _ = model(frames_t, atten_lim_db=12.0)
    np.testing.assert_allclose(np.asarray(case["model_out_frames"]), m12[0].numpy(), atol=1e-3)
    y12 = np.asarray(case["output_wav"], np.float32)
    np.testing.assert_allclose(y12[HOP:], off12[:-HOP], atol=1e-3)
    assert np.abs(y12 - y).max() > 1e-3  # the limit changes the output


def test_int8_model_runs(exported):
    import onnxruntime as ort

    out, manifest = exported
    assert manifest["onnx_int8"] == "ans_stream_int8.onnx"
    sess = ort.InferenceSession(str(out / manifest["onnx_int8"]), providers=["CPUExecutionProvider"])
    feeds = {
        "spec": np.ones((1, 1, 161, 2), np.float32),
        "state_in": np.zeros((2, 1, 256), np.float32),
        "atten_lim_db": np.array([100.0], np.float32),
    }
    for f in ("ans_stream_int8.onnx", "ans_stream_int8_static.onnx"):
        sess = ort.InferenceSession(str(out / f), providers=["CPUExecutionProvider"])
        o, s = sess.run(None, feeds)
        assert o.shape == (1, 1, 161, 2) and s.shape == (2, 1, 256)
    table = manifest["quantization"]["int8-static"]
    assert sum(c.get("qdq", 0) + c.get("int8", 0) for c in table.values()) > 0  # some ops run in int8
    assert table.get("GRU", {}).get("fp32", 0) >= 1  # the GRU stays fp32
