import os
import re
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

# -----------------------------
# CLEANUP HELPERS
# -----------------------------
STOPWORDS = {
    "and","or","with","the","a","an","of","for","to","from","in","on","by","as",
    "your","our","their","its","be","is","are","was","were","will","can","may",
    "beyond","very","more","less","that","this","those","these","while","during"
}

def clean_skill(skill):
    """Remove stopwords and invalid skill formats."""
    skill = skill.strip().lower()
    skill = re.sub(r"[^a-zA-Z0-9 +/#.-]", "", skill)

    # Remove 1-letter or fully useless items
    if len(skill) < 3:
        return None

    # Remove stopwords-only items
    if all(w in STOPWORDS for w in skill.split()):
        return None

    return skill


# -----------------------------
# AI SKILL EXTRACTION (IMPROVED)
# -----------------------------
def extract_skills_from_jd(jd_text):
    prompt = f"""
    Extract ONLY real skills from this Job Description.
    Return ONLY SKILLS — NOT phrases, NOT benefits, NOT responsibilities.

    Include:
    - Technical skills
    - Analytical skills
    - Soft skills
    - Tools / technologies
    - Software proficiency

    FORMAT STRICTLY LIKE THIS:
    skill1, skill2, skill3, skill4 ...

    Job Description:
    {jd_text}
    """

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
        
        # Split by comma
        items = [x.strip().lower() for x in raw.split(",") if x.strip()]

        cleaned = []
        for item in items:
            c = clean_skill(item)
            if c:
                cleaned.append(c)

        # Remove duplicates and long phrases
        final = []
        for s in cleaned:
            if len(s.split()) <= 3 and s not in final:
                final.append(s)

        return final

    except Exception:
        return fallback_extract(jd_text)


# -----------------------------
# FALLBACK EXTRACTOR (NO AI)
# -----------------------------
def fallback_extract(text):
    text = text.lower()

    possible_skills = set()

    # Extract capitalized technical terms
    capitals = re.findall(r"\b[A-Za-z]{2,}\b", text)
    for c in capitals:
        possible_skills.add(c.lower())

    # Extract words ending with these keywords
    for word in text.split():
        if any(k in word for k in ["excel", "sql", "analysis", "report", "macro"]):
            possible_skills.add(word.lower())

    # Final cleanup
    cleaned = []
    for p in possible_skills:
        c = clean_skill(p)
        if c:
            cleaned.append(c)

    return cleaned


# -----------------------------
# AI Resume Improvement
# -----------------------------
def generate_resume_suggestions(resume_text, job_description):
    prompt = f"""
    You are an expert resume reviewer.
    Compare the resume with the job description and provide
    5–7 short and actionable improvements.

    Resume:
    {resume_text[:4000]}

    Job Description:
    {job_description[:4000]}
    """

    try:
        response = model.generate_content(prompt)
        if response and hasattr(response, "text"):
            return response.text.strip()
        return "No suggestions available."
    except Exception as e:
        return f"⚠️ AI suggestion generation failed: {str(e)}"
