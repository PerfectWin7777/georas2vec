"""Model weights and the small learned models used by the pipeline.

They are downloaded once from the Hugging Face model repository and cached locally.
Set GEORAS2VEC_ASSETS to a folder that already contains the files to work fully offline.
"""
import os

REPO_ID = os.environ.get("GEORAS2VEC_REPO", "PerfectWin/georas2vec-model")
REVISION = os.environ.get("GEORAS2VEC_REVISION", "v0.7")
FILES = {
    "weights": "best_fp16.pt",          # segmentation network (affinities + faults)
    "naming": "naming_model.pkl",       # text -> polygon scorer
    "ranker": "legend_ranker_sw.pkl",   # (polygon, code) ranker, trained with real printed legends
    "code_reader": "code_reader.pt",    # reader of geological unit codes (Є, Ʀ, superscripts, small capitals)
    "swatch_encoder": "swatch_enc.pt",  # learned legend-swatch / polygon comparator (colour + pattern)
    "water": "water_model.pkl",         # water / outside-the-map regions
    "propagation": "prop_model.pkl",    # colour propagation decision model
}


def fetch_assets(folder=None):
    """Return {role: local path} for every file in FILES."""
    folder = folder or os.environ.get("GEORAS2VEC_ASSETS")
    if folder:
        out = {k: os.path.join(folder, v) for k, v in FILES.items()}
        missing = [p for p in out.values() if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"missing model files: {missing}")
        return out
    from huggingface_hub import hf_hub_download
    return {k: hf_hub_download(REPO_ID, v, revision=REVISION) for k, v in FILES.items()}
