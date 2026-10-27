# gen3d-worker

RunPod Serverless worker: image → textured GLB with TRELLIS.2 (MIT), built on the
commercial-use fork that renders with DRTK instead of nvdiffrast.

Licenses: TRELLIS.2 code and weights — MIT; GaXxO-dev/TRELLIS.2-commercial-use — MIT;
DRTK — MIT; CuMesh / FlexGEMM / o-voxel — MIT; BiRefNet — MIT; DINOv3 — DINOv3 License
(gated on Hugging Face, downloaded at runtime with `HF_TOKEN`, not included in the image).

Input: `{"image": "<base64 PNG>", "pipeline_type": "1024_cascade", "seed": 42,
"decimation_target": 150000, "texture_size": 1024}` or `{"ping": true}`.
Output: `{"glb_base64": "...", "glb_bytes": N, "timings": {...}, "gpu": "..."}`.
