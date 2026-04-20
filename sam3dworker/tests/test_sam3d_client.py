from __future__ import annotations

from pathlib import Path

from sam3dworker import Sam3dWorkerClient


def test_sam3d_worker_client_context_manager_reconstructs_example_payload(
    gpu_test_environment: dict[str, str],
    sam3d_run_root: Path,
    sam3d_test_payloads: list[dict[str, object]],
) -> None:
    payload = dict(sam3d_test_payloads[0])
    context_root = sam3d_run_root / "context-manager"
    context_root.mkdir(parents=True, exist_ok=True)
    payload["output_dir"] = str(context_root / str(payload["label"]))

    with Sam3dWorkerClient(
        socket_path=context_root / "sam3d-client.sock",
        trace_path=context_root / "sam3d-client-trace.jsonl",
        python_executable=gpu_test_environment["python_executable"],
        startup_timeout=900.0,
    ) as client:
        response = client.reconstruct(
            image_path=str(payload["image_path"]),
            depth_path=str(payload["depth_path"]),
            mask_path=str(payload["mask_path"]),
            output_dir=str(payload["output_dir"]),
            fx=float(payload["fx"]),
            fy=float(payload["fy"]),
            cx=float(payload["cx"]),
            cy=float(payload["cy"]),
            label=str(payload["label"]),
            artifact_types=list(payload.get("artifact_types", [])),
            request_id="sam3d-context-manager",
            timeout=1200.0,
        )

    assert response["worker"] == "sam3d"
    assert response["label"] == payload["label"]
    assert Path(str(response["pointmap_path"])).is_file()
    assert Path(str(response["pointmap_path"])).is_relative_to(sam3d_run_root)


def test_sam3d_worker_client_reconstructs_all_payloads_with_shared_worker(
    sam3d_reconstruct_responses: list[dict[str, object]],
) -> None:
    labels = [str(response["label"]) for response in sam3d_reconstruct_responses]
    assert labels == ["red_cube_0", "blue_cube_0"]
    assert all(response["worker"] == "sam3d" for response in sam3d_reconstruct_responses)
