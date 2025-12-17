#writefile realistic_upsc_mcq_pdfs.py
from fpdf import FPDF
import random
import os

# ----------------------------
# CONFIG
# ----------------------------
NUM_QUESTIONS = 500
OUTPUT_DIR = "upsc_realistic_mcq_pdfs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Subjects and keywords for generating realistic MCQs
SUBJECTS = {
    "Polity": ["constitution", "parliament", "judiciary", "rights", "emergency", "fundamental duties", "federalism"],
    "Sociology": ["society", "caste", "class", "gender", "social change", "social mobility", "community"],
    "Psychology": ["behavior", "motivation", "emotion", "learning", "memory", "personality", "cognition"],
    "Business Studies": ["management", "marketing", "planning", "leadership", "finance", "strategy", "HRM"],
    "Economics": ["GDP", "inflation", "growth", "development", "poverty", "unemployment", "fiscal policy"]
}

# ----------------------------
# FUNCTIONS
# ----------------------------
def generate_options(subject, correct_keyword):
    """Generate 4 options with 1 correct and 3 distractors."""
    keywords = SUBJECTS[subject]
    options = [correct_keyword]
    distractors = random.sample([k for k in keywords if k != correct_keyword], 3)
    options.extend(distractors)
    random.shuffle(options)
    correct_index = options.index(correct_keyword)
    return options, correct_index

def generate_mcq(subject, qnum):
    """Generate one UPSC-style MCQ."""
    correct_keyword = random.choice(SUBJECTS[subject])
    question = f"Q{qnum}. Which of the following best describes '{correct_keyword}' in {subject}?"
    options, answer_idx = generate_options(subject, correct_keyword)
    return question, options, answer_idx

def create_pdf(subject):
    """Create PDF file with MCQs for a subject."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Use built-in Arial font (no TTF required)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"UPSC-style Realistic MCQs - {subject}", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", '', 12)
    for i in range(1, NUM_QUESTIONS + 1):
        question, options, answer_idx = generate_mcq(subject, i)
        pdf.multi_cell(0, 8, question)
        for idx, opt in enumerate(options):
            pdf.cell(0, 8, f"{chr(97 + idx)}) {opt}", ln=True)
        pdf.cell(0, 8, f"Answer: {chr(97 + answer_idx)}) {options[answer_idx]}", ln=True)
        pdf.ln(2)

    file_path = os.path.join(OUTPUT_DIR, f"{subject}_mcqs.pdf")
    pdf.output(file_path)
    print(f"{subject} PDF generated: {file_path}")

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    for subject in SUBJECTS:
        create_pdf(subject)
    print("✅ All 5 UPSC-style realistic MCQ PDFs generated successfully.")

