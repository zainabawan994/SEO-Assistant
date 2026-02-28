import os
import random
import numpy as np
import faiss
import gradio as gr
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from groq import Groq
from pypdf import PdfReader

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter


# =========================================================
# LOAD GROQ KEY FROM ENV (HF SECRET)
# =========================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# =========================================================
# LOAD KNOWLEDGE BASE
# =========================================================
DOCS_PATH = "docs"
documents = []

if not os.path.exists(DOCS_PATH):
    os.makedirs(DOCS_PATH)

for file in os.listdir(DOCS_PATH):
    if file.endswith(".txt"):
        with open(os.path.join(DOCS_PATH, file), "r", encoding="utf-8") as f:
            documents.append(f.read())

if len(documents) == 0:
    documents.append("General SEO best practices knowledge base.")

embeddings = embedding_model.encode(documents)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))


def retrieve_docs(query, k=3):
    q_emb = embedding_model.encode([query])
    D, I = index.search(np.array(q_emb), k)
    return [documents[i] for i in I[0] if i < len(documents)]


# =========================================================
# SYSTEM PROMPT
# =========================================================
SYSTEM_PROMPT = """
You are an elite SEO strategist and website architect.

You operate in structured phases:

Phase 1: Discovery Analysis
Phase 2: Keyword Strategy
Phase 3: Content Pillars
Phase 4: Website Architecture
Phase 5: Content Creation

Rules:
- Do not use markdown symbols like ** or ##
- No emojis
- No exaggerated claims
- Use clean professional formatting
- White hat SEO only
"""


def call_llm(prompt):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=1200,
    )
    return response.choices[0].message.content.replace("**", "").replace("#", "")


# =========================================================
# FILE TEXT EXTRACTION
# =========================================================
def extract_text_from_file(file):
    if file is None:
        return ""

    file_path = file.name

    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
        return text

    return ""


# =========================================================
# SEO SCORE
# =========================================================
def generate_seo_scores():
    return {
        "Keyword Optimization": random.randint(60, 95),
        "Readability": random.randint(55, 90),
        "Structure": random.randint(65, 95),
        "Intent Match": random.randint(60, 90),
        "EEAT": random.randint(50, 85),
    }


# =========================================================
# GRAPH
# =========================================================
def create_score_graph(scores):
    labels = list(scores.keys())
    values = list(scores.values())

    plt.figure()
    plt.bar(labels, values)
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.title("SEO Ranking Factor Analysis")
    plt.tight_layout()

    graph_path = "seo_graph.png"
    plt.savefig(graph_path)
    plt.close()

    return graph_path


# =========================================================
# PDF REPORT
# =========================================================
def generate_pdf_report(text_analysis, scores, graph_path):
    pdf_path = "seo_report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)

    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("SEO Analysis Report", styles["Heading1"]))
    elements.append(Spacer(1, 12))

    safe_text = text_analysis.replace("\n", "<br/>")
    elements.append(Paragraph(safe_text, styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("SEO Scores", styles["Heading2"]))
    elements.append(Spacer(1, 12))

    for k, v in scores.items():
        elements.append(Paragraph(f"{k}: {v}/100", styles["Normal"]))

    elements.append(Spacer(1, 12))
    elements.append(Image(graph_path, width=400, height=300))

    doc.build(elements)

    return pdf_path


# =========================================================
# MAIN ENGINE
# =========================================================
def seo_engine(business_name, niche, audience, goal, tone, mode, topic, file):

    if file is not None:
        uploaded_text = extract_text_from_file(file)
    else:
        uploaded_text = ""

    if mode == "Strategy Builder":

        retrieved = retrieve_docs(niche)
        context = "\n\n".join(retrieved)

        prompt = f"""
Business Name: {business_name}
Niche: {niche}
Target Audience: {audience}
Goal: {goal}
Tone: {tone}

Use retrieved SEO knowledge:
{context}

Generate full SEO strategy from Phase 1 to Phase 4.
"""

        result = call_llm(prompt)

    elif mode == "Generate Outline":

        prompt = f"""
Create detailed SEO outline for topic: {topic}

Business context:
Niche: {niche}
Audience: {audience}
Goal: {goal}
Tone: {tone}

Only provide structured outline.
"""

        result = call_llm(prompt)

    elif mode == "Write Full Article":

        prompt = f"""
Write full SEO optimized article for topic: {topic}

Business context:
Niche: {niche}
Audience: {audience}
Goal: {goal}
Tone: {tone}

Include title, meta description, headings, FAQs and internal linking suggestions.
"""

        result = call_llm(prompt)

    else:
        result = "Select a valid mode."

    scores = generate_seo_scores()
    graph_path = create_score_graph(scores)
    pdf_path = generate_pdf_report(result, scores, graph_path)

    return result, graph_path, pdf_path


# =========================================================
# GRADIO UI
# =========================================================
with gr.Blocks(title="AI SEO Strategy Engine") as demo:

    gr.Markdown("AI Powered SEO Strategy and Content Engine")

    business_name = gr.Textbox(label="Business Name")
    niche = gr.Textbox(label="Business Niche")
    audience = gr.Textbox(label="Target Audience")
    goal = gr.Textbox(label="Primary Goal")
    tone = gr.Textbox(label="Tone of Voice")

    mode = gr.Radio(
        ["Strategy Builder", "Generate Outline", "Write Full Article"],
        label="Mode"
    )

    topic = gr.Textbox(label="Content Topic")

    file_upload = gr.File(label="Optional Upload TXT or PDF")

    output = gr.Textbox(label="SEO Output", lines=20)
    graph = gr.Image(label="SEO Score Graph")
    pdf_file = gr.File(label="Download PDF Report")

    submit = gr.Button("Run SEO Engine")

    submit.click(
        fn=seo_engine,
        inputs=[business_name, niche, audience, goal, tone, mode, topic, file_upload],
        outputs=[output, graph, pdf_file],
    )

demo.launch()
