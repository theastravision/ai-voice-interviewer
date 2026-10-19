"""Apply Hugging Face token from environment (gated Indic Parler model)."""


def apply_hf_auth() -> None:
    import os

    token = (os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not token:
        return
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
