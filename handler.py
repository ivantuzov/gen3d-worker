"""gen3d-worker: RunPod Serverless handler, TRELLIS.2 (коммерческий форк на DRTK) image -> GLB.

Вход (job.input):
  image            base64 PNG (желательно RGBA с уже снятым фоном; иначе фон снимет BiRefNet)
  pipeline_type    "512" | "1024" | "1024_cascade" (по умолч.) | "1536_cascade"
  seed             int (42)
  decimation_target  граней в экспортируемом GLB (150000; дальше всё равно simplify на клиенте)
  texture_size     1024 | 2048 (1024)
Выход: {glb_base64, glb_bytes, timings{load,gen,export}, gpu}

Env:
  HF_TOKEN   токен HF с одобренным доступом к facebook/dinov3-vitl16-pretrain-lvd1689m
             (+ в fine-grained токене включено «Read access to public gated repos»)
  TRELLIS2_LOW_VRAM  auto|0|1 (auto: включается при < 40 ГБ VRAM)
"""
import base64, io, json, os, tempfile, time, traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")

import runpod
import torch
from PIL import Image
from huggingface_hub import snapshot_download

MODEL_REPO = "microsoft/TRELLIS.2-4B"
DINO_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
REMBG_REPO = "ZhengPeng7/BiRefNet"      # MIT; оригинальный briaai/RMBG-2.0 — CC BY-NC
PIPELINE_CONFIG = "pipeline_gen3d.json"
MAX_FACES = 16_777_216                  # предел растеризатора
PIPELINE_TYPES = ("512", "1024", "1024_cascade", "1536_cascade")

_pipe, _err, _load_s = None, None, None


def _low_vram():
    v = os.environ.get("TRELLIS2_LOW_VRAM", "auto").lower()
    if v in ("1", "true"): return True
    if v in ("0", "false"): return False
    return torch.cuda.get_device_properties(0).total_memory / 2**30 < 40


def get_pipeline():
    global _pipe, _err, _load_s
    if _pipe is not None or _err is not None:
        return _pipe, _err
    t = time.time()
    try:
        model_dir = snapshot_download(MODEL_REPO)
        token = os.environ.get("HF_TOKEN") or None
        dino_dir = snapshot_download(DINO_REPO, token=token)  # 403 => доступ к DINOv3 не одобрен
        cfg = json.load(open(os.path.join(model_dir, "pipeline.json")))
        cfg["args"]["image_cond_model"]["args"]["model_name"] = dino_dir
        cfg["args"]["rembg_model"]["args"]["model_name"] = REMBG_REPO
        cfg["args"]["low_vram"] = _low_vram()
        # snapshot-каталог HF read-only по ссылкам: пишем конфиг рядом через копию-ссылку
        work = "/tmp/trellis2_model"
        if not os.path.isdir(work):
            os.makedirs(work)
            for name in os.listdir(model_dir):
                os.symlink(os.path.join(model_dir, name), os.path.join(work, name))
        json.dump(cfg, open(os.path.join(work, PIPELINE_CONFIG), "w"), indent=1)

        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        pipe = Trellis2ImageTo3DPipeline.from_pretrained(work, PIPELINE_CONFIG)
        if getattr(pipe, "rembg_model", None) is not None:
            pipe.rembg_model.model.float()   # BiRefNet приходит в fp16, обёртка шлёт fp32
        dino = pipe.image_cond_model.model   # transformers 5: слои в dino.model.layer
        if not hasattr(dino, "layer") and hasattr(getattr(dino, "model", None), "layer"):
            dino.__dict__["layer"] = dino.model.layer
        pipe.cuda()
        _pipe = pipe
    except Exception as e:
        traceback.print_exc()
        _err = f"{type(e).__name__}: {e}"
    _load_s = round(time.time() - t, 1)
    print(f"[gen3d] pipeline load {_load_s}s err={_err}", flush=True)
    return _pipe, _err


def handler(job):
    inp = job.get("input") or {}
    if inp.get("ping"):   # проверка воркера без генерации
        return {"pong": True, "loaded": _pipe is not None, "error": _err, "load_s": _load_s,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    if not inp.get("image"):
        return {"error": "image (base64) is required"}
    ptype = str(inp.get("pipeline_type", "1024_cascade"))
    if ptype not in PIPELINE_TYPES:
        return {"error": f"pipeline_type must be one of {PIPELINE_TYPES}"}
    seed = int(inp.get("seed", 42))
    dec = max(10_000, min(int(inp.get("decimation_target", 150_000)), 2_000_000))
    tex = int(inp.get("texture_size", 1024))
    if tex not in (1024, 2048, 4096):
        return {"error": "texture_size must be 1024|2048|4096"}
    try:
        raw = inp["image"].split(",", 1)[-1]
        img = Image.open(io.BytesIO(base64.b64decode(raw)))
        img.load()
    except Exception as e:
        return {"error": f"bad image: {e}"}

    first = _pipe is None
    pipe, err = get_pipeline()
    if pipe is None:
        return {"error": f"pipeline init failed: {err}"}
    try:
        import o_voxel
        t0 = time.time()
        mesh = pipe.run(img, seed=seed, pipeline_type=ptype)[0]
        mesh.simplify(MAX_FACES)
        t1 = time.time()
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
            coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=dec, texture_size=tex,
            remesh=True, remesh_band=1, remesh_project=0, verbose=False)
        del mesh
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.glb")
            glb.export(p, extension_webp=True)
            data = open(p, "rb").read()
        torch.cuda.empty_cache()
        return {"glb_base64": base64.b64encode(data).decode(), "glb_bytes": len(data),
                "timings": {"load": _load_s if first else 0, "gen": round(t1 - t0, 1),
                            "export": round(time.time() - t1, 1)},
                "gpu": torch.cuda.get_device_name(0), "pipeline_type": ptype}
    except Exception as e:
        traceback.print_exc()
        torch.cuda.empty_cache()
        return {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    get_pipeline()  # прогрев при старте контейнера
    runpod.serverless.start({"handler": handler})
