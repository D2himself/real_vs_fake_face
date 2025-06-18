from fastai.vision.all import *
from fastai.callback.mixup import MixUp
from fastai.metrics import F1Score, Precision, Recall
from huggingface_hub import hf_hub_download
import gradio as gr


model_path = hf_hub_download(
    repo_id="Omokemi/real-vs-ai-model",  
    filename="real_vs_ai_clean.pkl"      
)

learn = load_learner(model_path)

# 🧠 Inference function
def classify_image(img):
    pred_class, pred_idx, probs = learn.predict(img)
    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}

# 📷 Optional example images
examples = [
    'fake_face_1.jpg', 'fake_face_2 (1).jpg', 
    'fake_face_3.jpg', 'fake_face_4.jpg', 'real_face.jpg'
]

# 🎛️ Gradio interface
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    examples=examples,
    title="Real vs AI Face Classifier",
    description="Upload a face image to detect if it's real or AI-generated."
)

if __name__ == "__main__":
    demo.launch()

