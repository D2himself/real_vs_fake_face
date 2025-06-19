from fastai.vision.all import *
from fastai.callback.mixup import MixUp
from fastai.metrics import Precision, Recall, F1Score
from huggingface_hub import hf_hub_download
import gradio as gr

# Load model from Hugging Face Model Hub
model_path = hf_hub_download(
    repo_id="Omokemi/real-vs-ai-model",  # Change if needed
    filename="real_vs_ai_clean.pkl"
)

# Load the model — all custom classes are now in scope
learn = load_learner(model_path)

# Inference function
def classify_image(img):
    pred_class, pred_idx, probs = learn.predict(img)
    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}

# Optional example images (must exist in the repo)
examples = ["real_face.jpg", "fake_face_1.jpg"]

# Create Gradio UI
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    examples=examples,
    title="🧠 Real vs AI Face Classifier",
    description="Upload a face image to detect if it's real or AI-generated using a FastAI model."
)

# Run locally or in HF Space
if __name__ == "__main__":
    demo.launch()
