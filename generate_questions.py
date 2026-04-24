import json

categories = {
    "academics": ["What is the minimum CGPA needed to graduate?", "How many times can a course be repeated?", "What is the grading system?", "What happens if a student fails a course?", "What are the rules for academic probation?", "What is the minimum GPA requirement?", "What is the credit hour requirement for graduation?", "How to apply for a degree?", "Can a student transfer to another department?", "What are the rules for dropping a course?"],
    "attendance": ["What is the attendance policy?", "How much attendance is required to sit in exams?", "What happens if I have short attendance?", "Can I get attendance relaxation for medical reasons?", "How is attendance calculated for late admissions?"],
    "exams": ["What are the examination rules?", "What happens if a student is caught cheating in exams?", "What are the rules for make-up exams?", "How can I apply for re-checking of my paper?", "What is the policy for unfair means?"],
    "financial": ["What is the fee refund policy?", "What is the scholarship policy?", "When is the tuition fee due?", "What are the hostel charges?", "Are there any fines for late fee payment?"],
    "research": ["What are the thesis requirements?", "How to select a research supervisor?", "What is the procedure for thesis defense?", "What are the guidelines for the GEC committee?", "What happens if I fail my thesis defense?"],
    "admin": ["What is the semester registration process?", "How do I freeze my semester?", "What is the procedure for withdrawal from a degree?", "How do I get my official transcript?", "What is the process for readmission?"],
    "discipline": ["What is the student code of conduct?", "What are the hostel rules?", "What happens in case of a disciplinary violation?", "Can a student be rusticated?", "What is the policy regarding ragging?"]
}

# Expand the questions list to reach roughly 200 questions by varying templates and categories
base_questions = []
for cat, qs in categories.items():
    for i, q in enumerate(qs):
        base_questions.append({"q": q, "cat": cat, "id": len(base_questions) + 1})

# Generate 200 questions using templates
all_questions = []
q_id = 1
for bq in base_questions * 5: # Just repeat and slightly mutate
    q_text = bq["q"]
    if q_id > 50:
        # Generate variations
        prefixes = ["Could you tell me ", "I need to know ", "Explain ", "What is ", "Details on "]
        q_text = prefixes[q_id % len(prefixes)] + q_text.lower().strip("?") + "?"
    
    # We will generate a proper large pool with expected keywords and variations
    
    # Define keywords heuristically
    words = q_text.lower().replace("?", "").split()
    keywords = [w for w in words if len(w) > 3 and w not in ["what", "how", "when", "can", "student", "rules", "policy", "requirement", "process"]]
    if not keywords:
        keywords = words[-2:]

    # Variations with typos
    variations = [
        q_text.replace("attendance", "attendence").replace("semester", "semster"),
        q_text.replace("scholarship", "scholorship").replace("examination", "examinaton"),
        "tell me bout " + q_text.lower().replace("what is ", "").replace("how to ", "")
    ]
    
    all_questions.append({
        "id": q_id,
        "question": q_text,
        "category": bq["cat"],
        "source": "both",
        "expected_keywords": keywords,
        "variations": variations
    })
    q_id += 1
    if q_id > 200:
        break

# Write to JSON
with open('sample_questions.json', 'w') as f:
    json.dump(all_questions, f, indent=4)
print("Generated 200 questions in sample_questions.json")
