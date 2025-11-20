import re
from fpdf import FPDF
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load semantic model once
model = SentenceTransformer("all-MiniLM-L6-v2")


# Basic text cleanup
def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"[\r\n]", " ", text)
    text = re.sub(r"[^\x00-\x7F]", " ", text)  # remove non-ascii for stability
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Semantic similarity 0–100
def calculate_similarity(resume_text, jd_text):
    if not resume_text or not jd_text:
        return 0.0
    try:
        e = model.encode([resume_text, jd_text])
        score = cosine_similarity([e[0]], [e[1]])[0][0]
        return round(float(score) * 100, 2)
    except Exception:
        return 0.0


# Skill Matching
def find_missing_skills(resume_text, jd_text, extracted_skills):
    """
    extracted_skills = list of skills returned by extract_skills_from_jd()
    return: (missing_list, matched_list)
    """
    if not extracted_skills:
        return [], []

    resume_low = resume_text.lower()
    matched = []
    missing = []

    for skill in extracted_skills:
        sk = skill.lower().strip()

        # direct substring match
        if sk in resume_low:
            matched.append(sk)
            continue

        # token match
        parts = [p for p in re.split(r"[\s/,-]+", sk) if p]
        found = sum(1 for p in parts if p in resume_low)

        # require at least 50% + 1 words match
        if found >= max(1, len(parts) // 2 + 1):
            matched.append(sk)
        else:
            missing.append(sk)

    # remove duplicates
    def uniq(seq):
        seen = set()
        out = []
        for s in seq:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    return uniq(missing), uniq(matched)


# Resume structure heuristic (ATS scoring)
def resume_structure_score(resume_text):
    t = resume_text.lower()
    score = 0
    if "experience" in t or "work experience" in t:
        score += 35
    if "education" in t or "degree" in t:
        score += 25
    if "skills" in t:
        score += 20
    if "project" in t or "intern" in t:
        score += 20
    return min(score, 100)


# Experience alignment
def experience_alignment_score(resume_text, jd_text):
    r = resume_text.lower()
    j = jd_text.lower()
    score = 0

    # seniority check
    senior_words = ["senior", "lead", "manager", "jr.", "sr.", "junior", "associate"]
    if any(w in j for w in senior_words):
        if any(w in r for w in senior_words):
            score += 50
        else:
            score += 10

    # numeric experience
    req = re.search(r"(\d+)\+?\s+years", j)
    if req:
        req_years = int(req.group(1))
        got = re.search(r"(\d+)\s+years", r)
        if got:
            yr = int(got.group(1))
            if yr >= req_years:
                score += 50
            else:
                score += int(50 * yr / req_years)
        else:
            score += 10
    else:
        score += 50  # no experience requirement → neutral reward

    return min(score, 100)


# ATS Final Score (0–100)
def calculate_final_score(semantic_score, matched_skills, missing_skills, resume_text, jd_text):
    total = len(matched_skills) + len(missing_skills)
    skill_ratio = (len(matched_skills) / total) if total > 0 else 0

    skill_component = skill_ratio * 100
    structure = resume_structure_score(resume_text)
    experience = experience_alignment_score(resume_text, jd_text)

    final = (
        skill_component * 0.50 +
        semantic_score * 0.20 +
        structure * 0.15 +
        experience * 0.15
    )
    return round(final, 2)


# PDF generator
def sanitize_pdf_text(text):
    if not text:
        return ""
    text = text.replace("–", "-")  # en-dash
    text = text.replace("—", "-")  # em-dash
    t = re.sub(r'[^\x00-\x7F]+', '', text)
    return re.sub(r'\s+', ' ', t).strip()



def generate_pdf_report(score, matched, missing, ai_suggestions):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.set_auto_page_break(auto=True)

    today = datetime.now().strftime("%d %B %Y")

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(0, 70, 180)
    pdf.cell(0, 10, "AI Resume Analyzer Report", ln=True)

    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "ATS Optimized Report - By Parth Khatu", ln=True)

    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, f"Date: {today}", ln=True, align="R")
    pdf.ln(5)

    # Score
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Overall ATS Score", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(0, 8, sanitize_pdf_text(f"Your resume scored {score}%."))

    # Skills
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Matched Skills", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 8, sanitize_pdf_text(", ".join(matched) or "None"))

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Missing Skills", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 8, sanitize_pdf_text(", ".join(missing) or "None"))

    # AI Suggestions
    if ai_suggestions:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, "AI Suggestions", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 8, sanitize_pdf_text(ai_suggestions))

    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 10, sanitize_pdf_text("Generated by AI Resume Analyzer - ATS Mode"), align="C")


    return bytes(pdf.output(dest="S"))
