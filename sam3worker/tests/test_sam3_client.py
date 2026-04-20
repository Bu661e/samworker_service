from __future__ import annotations

from pathlib import Path

from sam3worker import Sam3WorkerClient


def test_sam3_worker_client_context_manager_example(
    gpu_test_environment: dict[str, str],
    sam3_test_inputs: dict[str, object],
    sam3_run_root: Path,
) -> None:
    output_dir = sam3_run_root / "context-manager" / "req-1"
    socket_path = sam3_run_root / "context-manager" / "sam3-client-example.sock"
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with Sam3WorkerClient(
        socket_path=socket_path,
        python_executable=gpu_test_environment["python_executable"],
        startup_timeout=180.0,
        trace_path=sam3_run_root / "context-manager" / "sam3-client-example-trace.jsonl",
    ) as client:
        response = client.infer(
            image_path=sam3_test_inputs["image_path"],
            output_dir=output_dir,
            bboxes=sam3_test_inputs["bboxes"],
            request_id="sam3-client-example",
            timeout=300.0,
        )

    assert response["worker"] == "sam3"
    assert response["image_path"] == str(sam3_test_inputs["image_path"])
    assert response["output_dir"] == str(output_dir)
    assert len(response["results"]) == len(sam3_test_inputs["bboxes"])


def test_sam3_worker_client_second_real_infer_run_succeeds(
    sam3_client: Sam3WorkerClient,
    sam3_test_inputs: dict[str, object],
    sam3_run_root: Path,
) -> None:
    output_dir = sam3_run_root / "second-run" / "req-2"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    response = sam3_client.infer(
        image_path=sam3_test_inputs["image_path"],
        output_dir=output_dir,
        bboxes=sam3_test_inputs["bboxes"],
        request_id="sam3-client-second-run",
        timeout=300.0,
    )

    assert response["worker"] == "sam3"
    assert response["batch_model_inference_ms"] > 0.0
    assert all(item["found"] is True for item in response["results"])
