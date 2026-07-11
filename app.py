"""
ClaimAI - AI Co-Pilot for Insurance Claim Processing
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import io
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from datetime import datetime
from fpdf import FPDF

MODEL_PATH = "best_model.pth"
LOG_FILE = "claims_log.csv"
IMG_SIZE = 64

st.set_page_config(page_title="ClaimAI Co-Pilot", page_icon="🚗", layout="wide")


class DamageCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 2)
        self.gradients = None
        self.activations = None

    def _save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        self.activations = x
        if x.requires_grad:
            x.register_hook(self._save_gradient)
        x = self.pool(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


@st.cache_resource
def load_model():
    model = DamageCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def preprocess_image(pil_img):
    img = pil_img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    tensor = torch.FloatTensor(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def generate_gradcam(model, input_tensor, class_idx):
    model.zero_grad()
    output = model(input_tensor)
    score = output[0, class_idx]
    score.backward()
    gradients = model.gradients[0]
    activations = model.activations[0].detach()
    weights = gradients.mean(dim=(1, 2))
    cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * activations[i]
    cam = F.relu(cam)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.numpy(), F.softmax(output, dim=1)[0].detach()


def overlay_heatmap(cam, original_pil_img, alpha=0.45):
    base_img = original_pil_img.resize((256, 256)).convert("RGB")
    base = np.array(base_img).astype(np.float32)
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize((256, 256), resample=Image.BILINEAR)
    cam_resized = np.array(cam_img).astype(np.float32) / 255.0
    r = np.clip(1.5 - np.abs(4 * cam_resized - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * cam_resized - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * cam_resized - 1), 0, 1)
    heat_color = np.stack([r, g, b], axis=-1) * 255.0
    overlay = base * (1 - alpha) + heat_color * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def compute_tabular_risk(vehicle_age, mileage, repair_cost, policy_type):
    risk = 0
    reasons = []
    if vehicle_age > 10:
        risk += 15; reasons.append("vehicle age over 10 years")
    if mileage > 150000:
        risk += 15; reasons.append("unusually high mileage")
    if repair_cost > 5000:
        risk += 25; reasons.append("high repair cost estimate")
    if policy_type == "Third-Party":
        risk += 10; reasons.append("third-party policy needs extra verification")
    return min(risk, 100), reasons


SUSPICIOUS_KEYWORDS = ["stolen", "total loss", "hit and run", "no witness",
                        "cash settlement", "unregistered", "no police report"]


def analyze_claim_text(text):
    text_lower = text.lower()
    flags = [kw for kw in SUSPICIOUS_KEYWORDS if kw in text_lower]
    risk = min(len(flags) * 25, 100)
    return risk, flags


def generate_explanation(damage_label, confidence, tabular_risk, tabular_reasons,
                          text_risk, text_flags, final_decision):
    parts = [f"The vision model classified the vehicle as **{damage_label}** with {confidence:.1%} confidence."]
    if tabular_reasons:
        parts.append("Policy/claim data raised these factors: " + "; ".join(tabular_reasons) + ".")
    else:
        parts.append("No unusual risk factors were found in the policy or vehicle data.")
    if text_flags:
        parts.append("The claim description contained flagged terms: " + ", ".join(text_flags) + ".")
    else:
        parts.append("No suspicious language was detected in the claim description.")
    parts.append(f"Combined assessment: **{final_decision}**.")
    return " ".join(parts)


def compute_final_decision(damage_label, confidence, tabular_risk, text_risk):
    if damage_label == "Whole" and tabular_risk < 30 and text_risk < 30:
        return "Auto-Approve (No Damage Detected)"
    if damage_label == "Damaged" and confidence > 0.85 and tabular_risk < 40 and text_risk < 20:
        return "Approve for Repair Payout"
    if tabular_risk >= 50 or text_risk >= 40:
        return "Flag for Manual Fraud Review"
    return "Route to Human Adjuster"


def log_decision(claim_id, damage_label, confidence, tabular_risk, text_risk,
                  final_decision, human_decision, notes=""):
    row = {
        "claim_id": claim_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ai_damage_label": damage_label,
        "ai_confidence": round(float(confidence), 3),
        "tabular_risk": tabular_risk,
        "text_risk": text_risk,
        "ai_recommendation": final_decision,
        "human_decision": human_decision,
        "notes": notes,
    }
    df_row = pd.DataFrame([row])
    if os.path.exists(LOG_FILE):
        df_row.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df_row.to_csv(LOG_FILE, mode="w", header=True, index=False)


def generate_pdf_report(claim_id, damage_label, confidence, tabular_risk, text_risk,
                         final_decision, explanation, human_decision):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    page_width = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.set_font("helvetica", "B", 16)
    pdf.set_x(pdf.l_margin)
    pdf.cell(page_width, 10, "AI Insurance Claim Assessment Report")
    pdf.ln(14)
    pdf.set_font("helvetica", "", 11)
    plain_explanation = explanation.replace("**", "")
    lines = [
        f"Claim ID: {claim_id}",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"AI Damage Assessment: {damage_label} ({confidence:.1%} confidence)",
        f"Tabular Risk Score: {tabular_risk}/100",
        f"Text Risk Score: {text_risk}/100",
        f"AI Recommendation: {final_decision}",
    ]
    for line in lines:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(page_width, 8, line)
    pdf.ln(4)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(page_width, 8, f"Explanation: {plain_explanation}")
    pdf.ln(4)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(page_width, 8, f"Human Reviewer Decision: {human_decision}")
    pdf_bytes = pdf.output(dest="S")
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode("latin-1")
    return bytes(pdf_bytes)


st.title("🚗 ClaimAI — Insurance Claim Co-Pilot")
st.caption("Multimodal AI assistant (image + tabular + text) for faster, explainable, human-supervised car insurance claim triage.")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found. Copy your trained best_model.pth into this folder.")
    st.stop()

model = load_model()
device = torch.device("cpu")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Upload Vehicle Photo")
    uploaded_file = st.file_uploader("Upload a photo of the vehicle", type=["jpg", "jpeg", "png"])

    st.subheader("2. Claim & Policy Details")
    vehicle_age = st.number_input("Vehicle age (years)", 0, 30, 5)
    mileage = st.number_input("Mileage (km)", 0, 500000, 40000, step=1000)
    repair_cost = st.number_input("Estimated repair cost ($)", 0, 50000, 1500, step=100)
    policy_type = st.selectbox("Policy type", ["Comprehensive", "Third-Party"])

    st.subheader("3. Claim Description")
    claim_text = st.text_area("Describe what happened", height=100)

    analyze_btn = st.button("🔍 Analyze Claim", type="primary")

with col2:
    st.subheader("AI Assessment")

    if analyze_btn:
        if uploaded_file is None:
            st.warning("Please upload a vehicle photo first.")
        else:
            image = Image.open(uploaded_file).convert("RGB")
            tensor = preprocess_image(image).to(device)
            output = model(tensor)
            probs = F.softmax(output, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())
            damage_label = "Damaged" if pred_idx == 1 else "Whole"
            confidence = float(probs[pred_idx].item())
            cam, _ = generate_gradcam(model, tensor, pred_idx)
            heatmap_img = overlay_heatmap(cam, image)
            tabular_risk, tabular_reasons = compute_tabular_risk(vehicle_age, mileage, repair_cost, policy_type)
            text_risk, text_flags = analyze_claim_text(claim_text)
            final_decision = compute_final_decision(damage_label, confidence, tabular_risk, text_risk)
            explanation = generate_explanation(damage_label, confidence, tabular_risk, tabular_reasons, text_risk, text_flags, final_decision)
            st.session_state["result"] = dict(damage_label=damage_label, confidence=confidence,
                tabular_risk=tabular_risk, tabular_reasons=tabular_reasons, text_risk=text_risk,
                text_flags=text_flags, final_decision=final_decision, explanation=explanation,
                heatmap_img=heatmap_img, original_image=image)

    if "result" in st.session_state:
        r = st.session_state["result"]
        c1, c2 = st.columns(2)
        with c1:
            st.image(r["original_image"], caption="Original photo")
        with c2:
            st.image(r["heatmap_img"], caption="Grad-CAM: where the AI looked")
        st.metric("Damage Prediction", r["damage_label"], f"{r['confidence']:.1%} confidence")
        cA, cB = st.columns(2)
        cA.metric("Tabular Risk Score", f"{r['tabular_risk']}/100")
        cB.metric("Text Risk Score", f"{r['text_risk']}/100")
        st.info(f"**AI Recommendation:** {r['final_decision']}")
        st.write("**AI Explanation:**")
        st.markdown(r["explanation"])

        st.subheader("4. Human-in-the-Loop Review")
        human_decision = st.radio("Adjuster decision", ["Approve", "Reject", "Modify / Escalate"], horizontal=True)
        notes = st.text_input("Reviewer notes (optional)")

        if "claim_id" not in st.session_state:
            st.session_state["claim_id"] = "CLM-" + datetime.now().strftime("%Y%m%d%H%M%S")
        claim_id = st.session_state["claim_id"]
        st.caption(f"Claim ID: {claim_id}")

        if st.button("✅ Submit Final Decision"):
            log_decision(claim_id, r["damage_label"], r["confidence"], r["tabular_risk"],
                r["text_risk"], r["final_decision"], human_decision, notes)
            st.success(f"Decision logged for {claim_id}")
            pdf_bytes = generate_pdf_report(claim_id, r["damage_label"], r["confidence"],
                r["tabular_risk"], r["text_risk"], r["final_decision"], r["explanation"], human_decision)
            st.download_button("📄 Download PDF Report", data=pdf_bytes,
                file_name=f"{claim_id}_report.pdf", mime="application/pdf")

st.divider()
st.subheader("📊 Claims Audit Log")
if os.path.exists(LOG_FILE):
    st.dataframe(pd.read_csv(LOG_FILE))
else:
    st.caption("No claims processed yet.")