from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import re
import tempfile
from PIL import Image
import pytesseract
import pdfplumber

app = Flask(
    __name__,
    static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static')
)

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# ---------- Encoding Maps (must match LabelEncoder from training) ----------
EDUCATION_MAP = {'Bachelors': 0, 'Masters': 1, 'PhD': 2}
UNIVERSITY_MAP = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
COMPANY_MAP = {'MNC': 0, 'Mid-size': 1, 'Startup': 2}


def count_items(text):
    """Count comma/space separated items in a text field."""
    if not text or not text.strip():
        return 0
    # Split by commas, semicolons, or newlines and count non-empty items
    items = [item.strip() for item in text.replace(';', ',').replace('\n', ',').split(',')]
    return len([item for item in items if item])


def compute_cri(data_dict):
    """
    Compute the Candidate Readiness Index (CRI) score (0–100).
    A weighted composite score reflecting overall candidate strength.
    """
    weights = {
        'cgpa': 15,            # max 10 → normalize to 0-15
        'internships': 10,     # max ~6 → normalize
        'projects': 10,
        'programming_languages': 10,
        'certifications': 10,
        'experience_years': 15,
        'hackathons': 5,
        'research_papers': 5,
        'skills_score': 10,    # already 0-30 range
        'soft_skills_score': 5,
        'education_level': 5,  # higher degree = more points
    }

    score = 0.0

    # CGPA (0-10 scale → 0-15 points)
    score += min(data_dict['cgpa'] / 10.0, 1.0) * weights['cgpa']

    # Internships (cap at 6)
    score += min(data_dict['internships'] / 6.0, 1.0) * weights['internships']

    # Projects (cap at 8)
    score += min(data_dict['projects'] / 8.0, 1.0) * weights['projects']

    # Programming languages (cap at 6)
    score += min(data_dict['programming_languages'] / 6.0, 1.0) * weights['programming_languages']

    # Certifications (cap at 6)
    score += min(data_dict['certifications'] / 6.0, 1.0) * weights['certifications']

    # Experience (cap at 10 years)
    score += min(data_dict['experience_years'] / 10.0, 1.0) * weights['experience_years']

    # Hackathons (cap at 5)
    score += min(data_dict['hackathons'] / 5.0, 1.0) * weights['hackathons']

    # Research papers (cap at 5)
    score += min(data_dict['research_papers'] / 5.0, 1.0) * weights['research_papers']

    # Skills score (0-30 → normalize)
    score += min(data_dict['skills_score'] / 30.0, 1.0) * weights['skills_score']

    # Soft skills score (0-10)
    score += min(data_dict['soft_skills_score'] / 10.0, 1.0) * weights['soft_skills_score']

    # Education level (Bachelors=0, Masters=1, PhD=2 → 0/2.5/5)
    score += (data_dict['education_level'] / 2.0) * weights['education_level']

    return round(score, 1)


def get_cri_label(cri):
    """Return a human-readable label for the CRI score."""
    if cri >= 80:
        return 'Excellent'
    elif cri >= 60:
        return 'Strong'
    elif cri >= 40:
        return 'Average'
    elif cri >= 20:
        return 'Below Average'
    else:
        return 'Weak'


def get_cri_color(cri):
    """Return a hex color for the CRI score."""
    if cri >= 80:
        return '#10b981'   # green
    elif cri >= 60:
        return '#10b981'   # green
    elif cri >= 40:
        return '#f59e0b'   # amber
    elif cri >= 20:
        return '#f97316'   # orange
    else:
        return '#ef4444'   # red


def get_dynamic_result(model_prediction, confidence, cri_score, cri_label):
    """
    Compute a dynamic hiring result that blends the ML model prediction
    with the CRI score.  Returns (result_text, result_status, result_tip).
    result_status is one of: 'hired', 'uncertain', 'rejected'
    """

    # ── CRI >= 80  (Excellent) ──────────────────────────────────────
    if cri_score >= 80:
        if model_prediction == 1:
            return (
                "Candidate is likely to be HIRED ✅",
                "hired",
                "Exceptional profile — strong across all dimensions."
            )
        else:
            return (
                "Strong profile but model predicts caution ⚠️",
                "uncertain",
                "CRI is excellent but the model flagged potential concerns. Review recommended."
            )

    # ── CRI 60-79  (Strong) ─────────────────────────────────────────
    elif cri_score >= 60:
        if model_prediction == 1:
            return (
                "Candidate is likely to be HIRED ✅",
                "hired",
                "Solid candidate — above-average readiness."
            )
        else:
            return (
                "Candidate may need further evaluation ⚠️",
                "uncertain",
                "Profile is strong but the model suggests further review."
            )

    # ── CRI 40-59  (Average) ────────────────────────────────────────
    elif cri_score >= 40:
        if model_prediction == 1:
            return (
                "Candidate may need further evaluation ⚠️",
                "uncertain",
                "Model leans positive but the CRI is only average. Proceed with caution."
            )
        else:
            return (
                "Candidate is likely to be REJECTED ❌",
                "rejected",
                "Average profile with a negative model prediction."
            )

    # ── CRI 20-39  (Below Average) ──────────────────────────────────
    elif cri_score >= 20:
        return (
            "Candidate is likely to be REJECTED ❌",
            "rejected",
            "Profile is below average — significant gaps detected."
        )

    # ── CRI < 20  (Weak) ────────────────────────────────────────────
    else:
        return (
            "Candidate is NOT suitable for hiring ❌",
            "rejected",
            "Very weak profile — does not meet minimum criteria."
        )


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # ---- Collect raw form data ----
    age = float(request.form.get('age', 0) or 0)
    cgpa = float(request.form.get('cgpa', 0) or 0)

    # Encoded categorical fields
    education_level = EDUCATION_MAP.get(request.form.get('education_level', 'Bachelors'), 0)
    university_tier = UNIVERSITY_MAP.get(request.form.get('university_tier', 'Tier 1'), 0)
    company_type = COMPANY_MAP.get(request.form.get('company_type', 'Startup'), 2)

    # Numeric fields
    internships = float(request.form.get('internships', 0) or 0)
    experience_years = float(request.form.get('experience_years', 0) or 0)
    research_papers = float(request.form.get('research_papers', 0) or 0)
    skills_score = float(request.form.get('skills_score', 0) or 0)
    resume_length_words = float(request.form.get('resume_length_words', 0) or 0)

    # TEXT inputs → count items
    programming_languages_text = request.form.get('programming_languages_text', '')
    certifications_text = request.form.get('certifications_text', '')
    projects_text = request.form.get('projects_text', '')
    hackathons_text = request.form.get('hackathons_text', '')
    soft_skills_text = request.form.get('soft_skills_text', '')

    programming_languages = count_items(programming_languages_text)
    certifications = count_items(certifications_text)
    projects = count_items(projects_text)
    hackathons = count_items(hackathons_text)

    # Soft skills score: use manual input if provided, else auto-calculate from text
    soft_skills_score_raw = request.form.get('soft_skills_score', '')
    if soft_skills_score_raw and soft_skills_score_raw.strip():
        soft_skills_score = float(soft_skills_score_raw)
    else:
        # Auto-calculate: count skills, cap at 10
        soft_skills_score = min(count_items(soft_skills_text), 10)

    # ---- Build feature array in EXACT training order ----
    # ['age', 'education_level', 'university_tier', 'cgpa', 'internships',
    #  'projects', 'programming_languages', 'certifications', 'experience_years',
    #  'hackathons', 'research_papers', 'skills_score', 'soft_skills_score',
    #  'resume_length_words', 'company_type']
    data = [
        age,
        education_level,
        university_tier,
        cgpa,
        internships,
        projects,
        programming_languages,
        certifications,
        experience_years,
        hackathons,
        research_papers,
        skills_score,
        soft_skills_score,
        resume_length_words,
        company_type
    ]

    # Scale and predict
    scaled_data = scaler.transform([data])
    prediction = model.predict(scaled_data)

    # Get prediction probability if available
    confidence = None
    try:
        proba = model.predict_proba(scaled_data)
        confidence = round(float(max(proba[0])) * 100, 1)
    except Exception:
        pass

    # Compute CRI score
    data_dict = {
        'age': age,
        'education_level': education_level,
        'university_tier': university_tier,
        'cgpa': cgpa,
        'internships': internships,
        'projects': projects,
        'programming_languages': programming_languages,
        'certifications': certifications,
        'experience_years': experience_years,
        'hackathons': hackathons,
        'research_papers': research_papers,
        'skills_score': skills_score,
        'soft_skills_score': soft_skills_score,
        'resume_length_words': resume_length_words,
        'company_type': company_type,
    }
    cri_score = compute_cri(data_dict)
    cri_label = get_cri_label(cri_score)
    cri_color = get_cri_color(cri_score)

    # ---- Dynamic result that links model + CRI ----
    result_text, result_status, result_tip = get_dynamic_result(
        prediction[0], confidence, cri_score, cri_label
    )

    return render_template(
        'index.html',
        prediction_text=result_text,
        result_status=result_status,
        result_tip=result_tip,
        confidence=confidence,
        cri_score=cri_score,
        cri_label=cri_label,
        cri_color=cri_color,
        # Pass form values back to preserve state
        form_data=request.form
    )


# ===================== Resume Scanner =====================

# Common programming languages for detection
KNOWN_LANGUAGES = [
    'python', 'java', 'javascript', 'c\+\+', 'c#', 'c', 'ruby', 'go', 'golang',
    'swift', 'kotlin', 'typescript', 'rust', 'scala', 'r', 'matlab', 'perl',
    'php', 'html', 'css', 'sql', 'bash', 'shell', 'dart', 'lua',
    'objective-c', 'assembly', 'haskell', 'elixir', 'clojure', 'julia'
]

KNOWN_SOFT_SKILLS = [
    'communication', 'leadership', 'teamwork', 'problem solving', 'problem-solving',
    'critical thinking', 'time management', 'adaptability', 'creativity',
    'collaboration', 'decision making', 'decision-making', 'negotiation',
    'presentation', 'public speaking', 'interpersonal', 'conflict resolution',
    'emotional intelligence', 'work ethic', 'flexibility', 'attention to detail',
    'organization', 'analytical', 'mentoring', 'coaching', 'empathy',
    'active listening', 'networking', 'multitasking', 'self-motivation'
]

KNOWN_CERTIFICATIONS_KW = [
    'aws', 'azure', 'gcp', 'google cloud', 'cisco', 'ccna', 'ccnp',
    'comptia', 'pmp', 'scrum', 'agile', 'itil', 'six sigma',
    'certified', 'certification', 'certificate', 'professional certificate',
    'data analytics', 'machine learning', 'deep learning', 'tensorflow',
    'coursera', 'udemy', 'edx', 'linkedin learning'
]


def extract_text_from_file(file):
    """Extract text from uploaded image or PDF file."""
    filename = file.filename.lower()
    
    if filename.endswith('.pdf'):
        # Save to temp file for pdfplumber
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        text = ''
        try:
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
        finally:
            os.unlink(tmp_path)
        return text
    
    else:
        # Image file — use OCR
        image = Image.open(file.stream)
        text = pytesseract.image_to_string(image)
        return text


def parse_resume_text(text):
    """Parse extracted resume text and return structured fields."""
    fields = {}
    text_lower = text.lower()
    lines = text.split('\n')
    
    # Word count (for resume_length_words)
    words = text.split()
    fields['resume_length_words'] = len(words)
    
    # --- Education Level ---
    if any(kw in text_lower for kw in ['ph.d', 'phd', 'doctorate', 'doctoral']):
        fields['education_level'] = 'PhD'
    elif any(kw in text_lower for kw in ['master', 'm.tech', 'mtech', 'm.sc', 'msc', 'm.s.', 'mba', 'm.b.a']):
        fields['education_level'] = 'Masters'
    elif any(kw in text_lower for kw in ['bachelor', 'b.tech', 'btech', 'b.sc', 'bsc', 'b.e.', 'b.s.', 'bca', 'bba']):
        fields['education_level'] = 'Bachelors'
    
    # --- CGPA ---
    cgpa_patterns = [
        r'(?:cgpa|gpa|cpi)\s*[:\-]?\s*(\d+\.?\d*)\s*(?:/\s*10)?',
        r'(\d\.\d{1,2})\s*(?:/\s*10\s*)?(?:cgpa|gpa|cpi)',
        r'(?:cgpa|gpa)\s*(\d\.\d{1,2})',
    ]
    for pattern in cgpa_patterns:
        match = re.search(pattern, text_lower)
        if match:
            val = float(match.group(1))
            if 0 < val <= 10:
                fields['cgpa'] = val
                break
    
    # --- Experience Years ---
    exp_patterns = [
        r'(\d+\.?\d*)\s*\+?\s*years?\s*(?:of\s+)?(?:experience|exp)',
        r'experience\s*[:\-]?\s*(\d+\.?\d*)\s*\+?\s*years?',
    ]
    for pattern in exp_patterns:
        match = re.search(pattern, text_lower)
        if match:
            fields['experience_years'] = float(match.group(1))
            break
    
    # --- Internships ---
    intern_patterns = [
        r'(\d+)\s*internships?',
        r'internships?\s*[:\-]?\s*(\d+)',
    ]
    for pattern in intern_patterns:
        match = re.search(pattern, text_lower)
        if match:
            fields['internships'] = int(match.group(1))
            break
    # Fallback: count occurrences of "intern" in different lines
    if 'internships' not in fields:
        intern_count = sum(1 for line in lines if re.search(r'\bintern\b', line.lower()))
        if intern_count > 0:
            fields['internships'] = intern_count
    
    # --- Programming Languages ---
    found_langs = []
    for lang in KNOWN_LANGUAGES:
        pattern = r'\b' + lang + r'\b'
        if re.search(pattern, text_lower):
            # Capitalize properly
            nice_name = lang.replace('\+\+', '++').replace('\#', '#').title()
            if lang == 'c\+\+':
                nice_name = 'C++'
            elif lang == 'c#':
                nice_name = 'C#'
            elif lang == 'c' and 'C++' in found_langs:
                continue  # skip bare 'c' if C++ already found
            elif lang == 'r':
                # Only match standalone R (avoid false positives)
                if not re.search(r'\bR\b', text):
                    continue
                nice_name = 'R'
            elif lang in ('html', 'css', 'sql', 'php'):
                nice_name = lang.upper()
            elif lang == 'javascript':
                nice_name = 'JavaScript'
            elif lang == 'typescript':
                nice_name = 'TypeScript'
            elif lang in ('golang', 'go'):
                nice_name = 'Go'
                if 'Go' in found_langs:
                    continue
            found_langs.append(nice_name)
    if found_langs:
        # Deduplicate
        fields['programming_languages_text'] = ', '.join(dict.fromkeys(found_langs))
    
    # --- Certifications ---
    found_certs = []
    for line in lines:
        line_lower = line.lower().strip()
        if any(kw in line_lower for kw in KNOWN_CERTIFICATIONS_KW):
            clean = line.strip()
            if clean and len(clean) > 3 and len(clean) < 150:
                found_certs.append(clean)
    if found_certs:
        fields['certifications_text'] = ', '.join(found_certs[:10])
    
    # --- Projects ---
    found_projects = []
    in_projects_section = False
    for line in lines:
        line_stripped = line.strip()
        line_lower = line_stripped.lower()
        if re.match(r'^(projects?|personal projects?|academic projects?|key projects?)', line_lower):
            in_projects_section = True
            continue
        elif in_projects_section:
            if re.match(r'^(experience|education|skills|certif|hackathon|achievement|award|hobby|interest|reference)', line_lower):
                in_projects_section = False
                continue
            if line_stripped and len(line_stripped) > 3:
                # Likely a project name/title
                project_name = re.sub(r'^[\-•●○▪▸►]\s*', '', line_stripped)
                if project_name and len(project_name) < 120:
                    found_projects.append(project_name)
    if found_projects:
        fields['projects_text'] = ', '.join(found_projects[:10])
    
    # --- Hackathons ---
    found_hackathons = []
    for line in lines:
        line_lower = line.lower()
        if 'hackathon' in line_lower or 'hackerathon' in line_lower or 'code jam' in line_lower or 'codejam' in line_lower:
            clean = line.strip()
            if clean and len(clean) > 3 and len(clean) < 150:
                found_hackathons.append(re.sub(r'^[\-•●○▪▸►]\s*', '', clean))
    if found_hackathons:
        fields['hackathons_text'] = ', '.join(found_hackathons[:10])
    
    # --- Soft Skills ---
    found_soft = []
    for skill in KNOWN_SOFT_SKILLS:
        if skill in text_lower:
            found_soft.append(skill.title())
    if found_soft:
        fields['soft_skills_text'] = ', '.join(found_soft)
        fields['soft_skills_score'] = min(len(found_soft), 10)
    
    # --- Research Papers ---
    paper_patterns = [
        r'(\d+)\s*(?:research\s+)?(?:papers?|publications?)',
        r'(?:papers?|publications?)\s*[:\-]?\s*(\d+)',
    ]
    for pattern in paper_patterns:
        match = re.search(pattern, text_lower)
        if match:
            fields['research_papers'] = int(match.group(1))
            break
    # Fallback: count lines with "published" or "paper" in them
    if 'research_papers' not in fields:
        pub_count = sum(1 for line in lines if re.search(r'\b(published|paper|publication|journal|conference)\b', line.lower()))
        if pub_count > 0:
            fields['research_papers'] = min(pub_count, 10)
    
    return fields


@app.route('/scan', methods=['POST'])
def scan_resume():
    """Handle resume file upload, extract text, parse fields, return JSON."""
    if 'resume' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded.'})
    
    file = request.files['resume']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected.'})
    
    try:
        text = extract_text_from_file(file)
        
        if not text or len(text.strip()) < 10:
            return jsonify({'success': False, 'error': 'Could not extract text from the file. Make sure the image is clear and readable.'})
        
        fields = parse_resume_text(text)
        return jsonify({'success': True, 'fields': fields})
    
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error processing file: {str(e)}'})


# ===================== Test Route =====================
@app.route('/test')
def test_cases():
    """Run worst / average / best test cases and display results."""
    cases = [
        {
            'name': '🔴 WORST — Weak Candidate',
            'data': {
                'age': 19, 'education_level': 0, 'university_tier': 2,
                'cgpa': 4.2, 'internships': 0, 'projects': 0,
                'programming_languages': 0, 'certifications': 0,
                'experience_years': 0, 'hackathons': 0,
                'research_papers': 0, 'skills_score': 3,
                'soft_skills_score': 1, 'resume_length_words': 80,
                'company_type': 0
            }
        },
        {
            'name': '🟠 BELOW AVERAGE — Gaps Detected',
            'data': {
                'age': 22, 'education_level': 0, 'university_tier': 2,
                'cgpa': 5.8, 'internships': 1, 'projects': 1,
                'programming_languages': 1, 'certifications': 0,
                'experience_years': 0.5, 'hackathons': 0,
                'research_papers': 0, 'skills_score': 8,
                'soft_skills_score': 3, 'resume_length_words': 200,
                'company_type': 2
            }
        },
        {
            'name': '🟡 AVERAGE — On the Fence',
            'data': {
                'age': 24, 'education_level': 0, 'university_tier': 1,
                'cgpa': 7.0, 'internships': 2, 'projects': 3,
                'programming_languages': 3, 'certifications': 1,
                'experience_years': 1.5, 'hackathons': 1,
                'research_papers': 0, 'skills_score': 15,
                'soft_skills_score': 5, 'resume_length_words': 400,
                'company_type': 1
            }
        },
        {
            'name': '🟢 STRONG — Likely Hire',
            'data': {
                'age': 26, 'education_level': 1, 'university_tier': 0,
                'cgpa': 8.5, 'internships': 4, 'projects': 5,
                'programming_languages': 5, 'certifications': 3,
                'experience_years': 4, 'hackathons': 3,
                'research_papers': 1, 'skills_score': 24,
                'soft_skills_score': 7, 'resume_length_words': 600,
                'company_type': 0
            }
        },
        {
            'name': '🏆 BEST — Top Candidate',
            'data': {
                'age': 28, 'education_level': 2, 'university_tier': 0,
                'cgpa': 9.6, 'internships': 6, 'projects': 8,
                'programming_languages': 6, 'certifications': 6,
                'experience_years': 8, 'hackathons': 5,
                'research_papers': 4, 'skills_score': 28,
                'soft_skills_score': 9, 'resume_length_words': 800,
                'company_type': 0
            }
        },
    ]

    results = []
    for case in cases:
        d = case['data']
        feature_arr = [
            d['age'], d['education_level'], d['university_tier'],
            d['cgpa'], d['internships'], d['projects'],
            d['programming_languages'], d['certifications'],
            d['experience_years'], d['hackathons'],
            d['research_papers'], d['skills_score'],
            d['soft_skills_score'], d['resume_length_words'],
            d['company_type']
        ]
        scaled = scaler.transform([feature_arr])
        pred = model.predict(scaled)
        conf = None
        try:
            proba = model.predict_proba(scaled)
            conf = round(float(max(proba[0])) * 100, 1)
        except Exception:
            pass

        cri = compute_cri(d)
        label = get_cri_label(cri)
        color = get_cri_color(cri)
        result_text, result_status, result_tip = get_dynamic_result(
            pred[0], conf, cri, label
        )

        results.append({
            'name': case['name'],
            'model_pred': 'HIRED' if pred[0] == 1 else 'REJECTED',
            'confidence': conf,
            'cri_score': cri,
            'cri_label': label,
            'cri_color': color,
            'result_text': result_text,
            'result_status': result_status,
            'result_tip': result_tip,
        })

    return render_template('test_results.html', results=results)


if __name__ == "__main__":
    app.run(debug=True)