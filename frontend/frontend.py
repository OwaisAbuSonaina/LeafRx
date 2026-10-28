import io
import os
import random
import gradio as gr
from PIL import Image
import requests

API_URL = os.getenv("API_URL", "http://0.0.0.0:8000/predict")
TEST_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "test_images")
VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def get_random_samples(k=3):
    """Fetch up to k random image paths from the test_images directory."""
    if not os.path.isdir(TEST_IMAGES_DIR):
        return []
    all_files = [
        os.path.join(TEST_IMAGES_DIR, f)
        for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith(VALID_EXTS)
    ]
    if not all_files:
        return []
    return random.sample(all_files, min(k, len(all_files)))


CUSTOM_CSS = """
#main-container {
    max-width: 980px;
    margin: 0 auto;
    padding: 1.5rem;
}
.header-box {
    text-align: center;
    padding: 1.8rem 1rem;
    border-radius: 14px;
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    margin-bottom: 1.5rem;
}
.header-box h1 {
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary-600);
    margin-bottom: 0.25rem;
}
.header-box p {
    color: var(--body-text-color-subdued);
    font-size: 0.95rem;
}
.result-card {
    border-radius: 12px;
    padding: 1rem 1.25rem;
    background: var(--background-fill-primary);
    border: 1px solid var(--border-color-primary);
}
.badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.85rem;
    margin-bottom: 12px;
}
.badge-healthy {
    background-color: rgba(34, 197, 94, 0.18);
    color: #16a34a;
    border: 1px solid rgba(34, 197, 94, 0.35);
}
.badge-disease {
    background-color: rgba(239, 68, 68, 0.18);
    color: #dc2626;
    border: 1px solid rgba(239, 68, 68, 0.35);
}
.plant-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--body-text-color);
    margin-bottom: 4px;
}
.condition-text {
    font-size: 1.05rem;
    color: var(--body-text-color);
    margin-bottom: 10px;
}
.meta-code {
    font-size: 0.85rem;
    color: var(--body-text-color-subdued);
}
.meta-code code {
    background: var(--background-fill-secondary);
    color: var(--body-text-color);
    padding: 2px 6px;
    border-radius: 4px;
}
.empty-placeholder {
    text-align: center;
    color: var(--body-text-color-subdued);
    padding: 2.2rem 1rem;
}
.samples-container {
    margin-top: 1rem;
    padding: 0.75rem;
    border-radius: 10px;
    background: var(--background-fill-secondary);
    border: 1px dashed var(--border-color-primary);
}
"""


def parse_label(raw_name: str):
    if "___" in raw_name:
        plant, disease = raw_name.split("___", 1)
    else:
        plant, disease = "Unknown", raw_name

    plant_clean = plant.replace("_", " ").strip()
    disease_clean = disease.replace("_", " ").strip()
    is_healthy = "healthy" in disease.lower()
    return plant_clean, disease_clean, is_healthy


def predict_leaf(image):
    if image is None:
        raise gr.Error("Please upload or select an image first.")

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    try:
        files = {"file": ("image.jpg", buffer, "image/jpeg")}
        response = requests.post(API_URL, files=files, timeout=20)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        raise gr.Error(f"Could not reach backend API at {API_URL}: {e}")

    raw_class = data.get("class", "Unknown")
    confidence = float(data.get("confidence", 0.0))

    plant, condition, is_healthy = parse_label(raw_class)

    badge_class = "badge-healthy" if is_healthy else "badge-disease"
    badge_text = "HEALTHY CROP" if is_healthy else "DISEASE DETECTED"

    summary_html = f"""
    <div class="result-card">
        <div class="badge {badge_class}">
            {badge_text}
        </div>
        <div class="plant-title">
            {plant}
        </div>
        <div class="condition-text">
            Condition: <b>{condition}</b>
        </div>
        <div class="meta-code">
            Model Raw Class: <code>{raw_class}</code>
        </div>
    </div>
    """

    label_distribution = {
        condition: confidence,
        "Other / Uncertainty": round(max(0.0, 1.0 - confidence), 4),
    }

    return summary_html, label_distribution


theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.green,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    block_radius="10px",
    block_border_width="1px",
)

with gr.Blocks(title="LeafRx - Crop Health AI") as demo:
    with gr.Column(elem_id="main-container"):
        gr.HTML(
            """
            <div class="header-box">
                <h1>🌿 LeafRx</h1>
                <p>AI-Powered Plant Disease Diagnostic System | Powered by ConvNeXt</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=5):
                image_input = gr.Image(
                    type="pil",
                    label="Leaf Photo",
                    sources=["upload", "clipboard"],
                    elem_classes=["result-card"],
                )
                with gr.Row():
                    clear_btn = gr.ClearButton(
                        components=[image_input], value="Clear"
                    )
                    submit_btn = gr.Button("Analyze Leaf", variant="primary")

                with gr.Column(elem_classes=["samples-container"]):
                    with gr.Row():
                        gr.Markdown("#### 🧪 Test Samples *(Click any image to test)*")
                        reroll_btn = gr.Button("🎲 Load 3 New Samples", size="sm")

                    gallery = gr.Gallery(
                        value=get_random_samples(3),
                        columns=3,
                        rows=1,
                        height=140,
                        show_label=False,
                        allow_preview=False,
                        object_fit="cover",
                    )

            with gr.Column(scale=5):
                result_html = gr.HTML(
                    value="""
                    <div class="result-card empty-placeholder">
                        Upload a clear leaf photo or click a sample to view diagnostic results.
                    </div>
                    """,
                )
                confidence_bar = gr.Label(
                    label="Prediction Confidence",
                    num_top_classes=2,
                    elem_classes=["result-card"],
                )

        # Re-roll samples into the gallery
        reroll_btn.click(
            fn=lambda: get_random_samples(3),
            inputs=[],
            outputs=[gallery],
        )

        # Selecting an image in the gallery loads it into image_input
        def on_sample_select(evt: gr.SelectData):
            # evt.value contains either the filepath string or a dict with {'image': {'path': ...}}
            if isinstance(evt.value, dict):
                path = evt.value.get("image", {}).get("path") or evt.value.get("image")
            else:
                path = evt.value
            return Image.open(path) if path else None

        gallery.select(fn=on_sample_select, outputs=[image_input])

        # Prediction handler
        submit_btn.click(
            fn=predict_leaf,
            inputs=[image_input],
            outputs=[result_html, confidence_bar],
        )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        theme=theme,
        css=CUSTOM_CSS,
    )