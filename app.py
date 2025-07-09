import streamlit as st
from transformers import pipeline
import re
import torch

@st.cache_resource
def load_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=device
    )


generator = load_model()

CLO_MAPPING = {
    "CLO 1-Knowledge": "Recall of facts, concepts, and basic principles. (Define, List, Identify, Recall)",
    "CLO 2-Comprehension": "Understand the meaning of information and interpret facts. (Explain, Describe, Summarize, Interpret)",
    "CLO 3-Application": "Use knowledge to solve problems or carry out tasks. (Apply, Demonstrate, Use, Solve)",
    "CLO 4-Analysis": "Break information into parts and understand structure/relationships. (Analyze, Compare, Contrast, Differentiate)",
    "CLO 5-Synthesis": "Compile information in new ways or propose alternatives. (Design, Develop, Compose, Formulate)",
    "CLO 6-Evaluation": "Make judgments based on criteria and standards. (Evaluate, Justify, Critique, Assess)"
}

BLOOM_VERBS = {
    "CLO 1-Knowledge": ["define", "list", "identify", "recall"],
    "CLO 2-Comprehension": ["explain", "describe", "summarize", "interpret"],
    "CLO 3-Application": ["apply", "demonstrate", "use", "solve"],
    "CLO 4-Analysis": ["analyze", "compare", "contrast", "differentiate"],
    "CLO 5-Synthesis": ["design", "develop", "compose", "formulate"],
    "CLO 6-Evaluation": ["evaluate", "justify", "critique", "assess"]
}

def generate_clos_and_skills(text):
    prompt = f"""
Extract detailed Course Learning Outcomes (CLOs) using action verbs from Bloom's Taxonomy (e.g., define, analyze, design, evaluate).
Also list relevant skill sets clearly as bullet points.

Course content:
{text}

Example CLOs:
1. Define the basic concepts of neural networks.
2. Explain different types of deep learning layers.
3. Apply deep learning models to real-world problems.
4. Analyze the performance of neural networks.
5. Design novel architectures for specific tasks.

Now generate the CLOs and skill sets:
"""
    output = generator(prompt, max_length=256, do_sample=False)[0]["generated_text"]
    return output

def extract_sentences(text):
    # Split text into sentences by ., !, ?
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences

def map_content_to_clos(content):
    sentences = extract_sentences(content)
    mapping = {}

    for clo_key, verbs in BLOOM_VERBS.items():
        matched_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(verb in sentence_lower for verb in verbs):
                matched_sentences.append(sentence.strip())
        if matched_sentences:
            mapping[clo_key] = matched_sentences
    return mapping

# Streamlit UI
st.title("📘 CLO & Skill Set Generator from Course Content")
st.markdown(
    "Upload your course content to automatically generate **CLOs**, **skill sets**, and "
    "**map course content to CLOs** based on Bloom’s Taxonomy."
)

uploaded_file = st.file_uploader("Upload Course Content (.txt)", type=["txt"])

if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    st.subheader("📄 Course Content Preview")
    st.write(content[:1000] + "..." if len(content) > 1000 else content)

    with st.spinner("Generating CLOs and Skill Sets..."):
        generated = generate_clos_and_skills(content)

    st.subheader("🎯 Generated CLOs and Skill Sets")
    st.markdown(generated.replace("\n", "  \n"))  # format line breaks nicely

    st.subheader("📌 Mapping of Course Content to CLOs (Bloom’s Taxonomy)")
    mapped = map_content_to_clos(content)

    if mapped:
        for clo_key, sentences in mapped.items():
            st.markdown(f"### {clo_key}: {CLO_MAPPING[clo_key]}")
            for s in sentences:
                st.write(f"- {s}")
    else:
        st.warning("No parts of the content matched CLO verbs.")
