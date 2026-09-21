import modal

app = modal.App("xai-hdetect-api")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi",
        "uvicorn[standard]",
        "python-multipart",
        "transformers>=4.37.0",
        "torch>=2.0.0",
        "torchvision",
        "Pillow",
        "numpy",
        "opencv-python-headless",
        "joblib",
        "lightgbm",
        "sentence-transformers",
        "huggingface_hub",
        "shap",
        "scikit-learn",
        "accelerate",
    )
    .add_local_file("main.py", "/root/main.py")
)

@app.function(
    image=image,
    gpu="A100",
    timeout=300,
    secrets=[modal.Secret.from_name("xai-hdetect-secrets")],
    scaledown_window=60,
)
@modal.asgi_app()
def fastapi_app():
    from main import app as fastapi_app
    return fastapi_app