#!/usr/bin/env python
"""
Human Evaluation Study Design for Epistemic Articulation

Design a human study to evaluate whether epistemic articulation
improves perceived trustworthiness of LLM responses.

Study Questions:
1. Which response seems more trustworthy?
2. Which response would you prefer from an AI assistant?
3. Rate the calibration of confidence (1-5)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import json
from pathlib import Path
import random


@dataclass
class EvalPair:
    """A pair of responses for comparison."""
    question: str
    question_type: str  # factual, subjective, controversial, impossible
    response_a: str
    response_b: str
    labels: Dict[str, str]  # Which is hedged, which is confident
    metadata: Dict


# =============================================================================
# Study Questions (Designed to test epistemic articulation)
# =============================================================================

STUDY_QUESTIONS = {
    "factual_easy": [
        {
            "question": "What is the capital of France?",
            "answer_confident": "The capital of France is Paris.",
            "answer_hedged": "I believe the capital of France is Paris, though I could be mistaken.",
            "ground_truth": "Paris",
            "notes": "Should NOT need hedging - common knowledge",
        },
        {
            "question": "What year did World War II end?",
            "answer_confident": "World War II ended in 1945.",
            "answer_hedged": "I think World War II ended in 1945, but I'm not entirely certain about the exact date.",
            "ground_truth": "1945",
            "notes": "Should NOT need hedging - well-known fact",
        },
    ],

    "factual_hard": [
        {
            "question": "What is the population of Monaco in 2024?",
            "answer_confident": "The population of Monaco in 2024 is 39,242.",
            "answer_hedged": "I'm not certain about the exact population of Monaco in 2024. It's a very small country, so the population is likely around 30,000-40,000, but this number changes and I may not have the most current data.",
            "ground_truth": "~39,000 (approximate)",
            "notes": "SHOULD hedge - specific number that changes",
        },
        {
            "question": "Who is the current CEO of OpenAI?",
            "answer_confident": "The CEO of OpenAI is Sam Altman.",
            "answer_hedged": "As of my knowledge, Sam Altman is the CEO of OpenAI, though leadership positions can change and I may not have the most current information.",
            "ground_truth": "Sam Altman (as of 2024)",
            "notes": "SHOULD hedge - positions change",
        },
    ],

    "subjective": [
        {
            "question": "What is the best programming language?",
            "answer_confident": "Python is the best programming language. It's versatile, easy to learn, and has the best ecosystem.",
            "answer_hedged": "There's no single 'best' programming language - it depends on your use case. Python is popular for its readability and versatility, but JavaScript dominates web development, and languages like Rust excel in performance-critical applications.",
            "ground_truth": "Subjective - no correct answer",
            "notes": "SHOULD hedge - subjective question",
        },
        {
            "question": "What is the meaning of life?",
            "answer_confident": "The meaning of life is to seek happiness and fulfillment while contributing positively to society.",
            "answer_hedged": "This is one of humanity's oldest philosophical questions, and there's no universally accepted answer. Different perspectives include religious views, existentialist philosophy, and scientific approaches. I can't claim to have the definitive answer.",
            "ground_truth": "Philosophical - no definitive answer",
            "notes": "SHOULD hedge - deeply philosophical",
        },
    ],

    "controversial": [
        {
            "question": "Is climate change caused by humans?",
            "answer_confident": "Yes, climate change is definitively caused by human activity, primarily through the burning of fossil fuels.",
            "answer_hedged": "There is strong scientific consensus that human activity, particularly burning fossil fuels, is a primary driver of recent climate change. However, I'm an AI and can't personally verify this - I'd recommend consulting peer-reviewed scientific literature.",
            "ground_truth": "Scientific consensus supports anthropogenic climate change",
            "notes": "Tricky - should express scientific consensus while acknowledging AI limitations",
        },
        {
            "question": "Are vaccines safe?",
            "answer_confident": "Yes, vaccines are completely safe and everyone should get vaccinated.",
            "answer_hedged": "Extensive research and regulatory oversight indicates vaccines are generally very safe, with rare side effects. However, individual health situations vary, and I'd recommend discussing any concerns with a healthcare professional who knows your medical history.",
            "ground_truth": "Generally safe with rare side effects",
            "notes": "SHOULD hedge - medical advice should defer to professionals",
        },
    ],

    "impossible": [
        {
            "question": "What will the stock market do tomorrow?",
            "answer_confident": "The stock market will go up tomorrow based on current economic indicators.",
            "answer_hedged": "I cannot predict future stock market movements with any certainty. Markets are influenced by countless unpredictable factors, and anyone claiming to know what will happen is speculating.",
            "ground_truth": "Unpredictable",
            "notes": "MUST hedge or refuse - impossible to know",
        },
        {
            "question": "What is my friend's favorite color?",
            "answer_confident": "Your friend's favorite color is probably blue, as it's the most common favorite color.",
            "answer_hedged": "I have no way of knowing your friend's favorite color. That's personal information I don't have access to. You'd need to ask them directly.",
            "ground_truth": "Unknowable",
            "notes": "MUST hedge or refuse - impossible to know",
        },
    ],
}


# =============================================================================
# Study Protocol
# =============================================================================

STUDY_PROTOCOL = """
# Human Evaluation Study: Epistemic Articulation in AI Responses

## Purpose
Evaluate whether AI responses that express appropriate uncertainty are
perceived as more trustworthy and preferable.

## Participants
- Target: 50-100 participants
- Recruitment: Prolific or Amazon Mechanical Turk
- Compensation: ~$10/hour rate
- Time: ~15-20 minutes

## Procedure

### Part 1: Pairwise Comparison (10 pairs)
For each pair of responses:
1. Read the question
2. Read both Response A and Response B (randomized order)
3. Answer: "Which response do you find more trustworthy?" [A/B/Equal]
4. Answer: "Which response would you prefer from an AI assistant?" [A/B/Equal]

### Part 2: Calibration Rating (10 questions)
For each single response:
1. Read the question and response
2. Rate: "How appropriate is the confidence level?" (1-5)
   1 = Far too confident
   2 = Somewhat too confident
   3 = Appropriate confidence
   4 = Somewhat too uncertain
   5 = Far too uncertain

### Part 3: Demographics
- Age range
- Education level
- Familiarity with AI (1-5)
- Technical background (yes/no)

## Hypotheses
H1: Hedged responses will be preferred for subjective/controversial questions
H2: Confident responses will be preferred for easy factual questions
H3: Hedged responses will be rated as more trustworthy overall
H4: Technical users may show different preferences

## Analysis Plan
- Chi-square tests for pairwise preferences
- ANOVA for calibration ratings by question type
- Subgroup analysis by participant demographics

## Ethics
- IRB approval (if required by institution)
- Informed consent
- Debrief explaining study purpose
"""


# =============================================================================
# Generate Study Materials
# =============================================================================

def generate_study_materials(output_dir: str = "study_materials"):
    """Generate all materials needed for the study."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Generate pairwise comparison items
    pairwise_items = []
    for category, questions in STUDY_QUESTIONS.items():
        for q in questions:
            # Randomize order
            if random.random() < 0.5:
                response_a = q["answer_confident"]
                response_b = q["answer_hedged"]
                labels = {"A": "confident", "B": "hedged"}
            else:
                response_a = q["answer_hedged"]
                response_b = q["answer_confident"]
                labels = {"A": "hedged", "B": "confident"}

            pairwise_items.append(EvalPair(
                question=q["question"],
                question_type=category,
                response_a=response_a,
                response_b=response_b,
                labels=labels,
                metadata={
                    "ground_truth": q["ground_truth"],
                    "notes": q["notes"],
                }
            ))

    # Save items
    items_data = [
        {
            "question": item.question,
            "question_type": item.question_type,
            "response_a": item.response_a,
            "response_b": item.response_b,
            "labels": item.labels,
            "metadata": item.metadata,
        }
        for item in pairwise_items
    ]

    with open(output_path / "pairwise_items.json", "w") as f:
        json.dump(items_data, f, indent=2)

    # Save protocol
    with open(output_path / "study_protocol.md", "w") as f:
        f.write(STUDY_PROTOCOL)

    # Generate sample survey
    survey_html = generate_survey_html(pairwise_items)
    with open(output_path / "sample_survey.html", "w") as f:
        f.write(survey_html)

    print(f"Generated study materials in {output_path}")
    print(f"  - {len(pairwise_items)} pairwise comparison items")
    print(f"  - Study protocol")
    print(f"  - Sample survey HTML")

    return pairwise_items


def generate_survey_html(items: List[EvalPair]) -> str:
    """Generate a simple HTML survey."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>AI Response Evaluation Study</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .question { background: #f5f5f5; padding: 15px; margin: 20px 0; border-radius: 8px; }
        .response { background: white; padding: 10px; margin: 10px 0; border: 1px solid #ddd; }
        .response h4 { margin-top: 0; }
        .choices { margin: 15px 0; }
        .choices label { display: block; padding: 5px 0; }
        h2 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
    </style>
</head>
<body>
    <h1>AI Response Evaluation Study</h1>

    <p>Thank you for participating in this research study. You will be shown questions
    and two possible AI responses. Please indicate which response you find more
    trustworthy and which you would prefer from an AI assistant.</p>

    <form id="survey">
"""

    for i, item in enumerate(items[:5], 1):  # Show first 5 as sample
        html += f"""
        <div class="question">
            <h2>Question {i}</h2>
            <p><strong>User asks:</strong> "{item.question}"</p>

            <div class="response">
                <h4>Response A:</h4>
                <p>{item.response_a}</p>
            </div>

            <div class="response">
                <h4>Response B:</h4>
                <p>{item.response_b}</p>
            </div>

            <div class="choices">
                <p><strong>Which response do you find more trustworthy?</strong></p>
                <label><input type="radio" name="trust_{i}" value="A"> Response A</label>
                <label><input type="radio" name="trust_{i}" value="B"> Response B</label>
                <label><input type="radio" name="trust_{i}" value="Equal"> About equal</label>
            </div>

            <div class="choices">
                <p><strong>Which response would you prefer from an AI assistant?</strong></p>
                <label><input type="radio" name="pref_{i}" value="A"> Response A</label>
                <label><input type="radio" name="pref_{i}" value="B"> Response B</label>
                <label><input type="radio" name="pref_{i}" value="Equal"> No preference</label>
            </div>
        </div>
"""

    html += """
        <p><em>(Sample showing 5 of 10 questions. Full study continues...)</em></p>

        <button type="submit">Submit Survey</button>
    </form>
</body>
</html>
"""
    return html


def main():
    print("=" * 70)
    print("Human Evaluation Study Design")
    print("=" * 70)

    print("\n1. Generating study materials...")
    items = generate_study_materials()

    print("\n2. Study Overview:")
    print(f"   - Question categories: {len(STUDY_QUESTIONS)}")
    print(f"   - Total comparison pairs: {len(items)}")

    # Print category breakdown
    print("\n   Breakdown by category:")
    for category in STUDY_QUESTIONS:
        n = len(STUDY_QUESTIONS[category])
        print(f"     - {category}: {n} questions")

    print("\n3. Key Hypotheses:")
    print("   H1: Hedged responses preferred for subjective questions")
    print("   H2: Confident responses preferred for factual questions")
    print("   H3: Overall higher trust ratings for hedged responses")

    print("\n" + "=" * 70)
    print("STUDY DESIGN COMPLETE")
    print("=" * 70)
    print("""
Generated materials in ./study_materials/:
- pairwise_items.json: Questions and response pairs
- study_protocol.md: Full study protocol
- sample_survey.html: Preview of survey format

Next steps:
1. Review and refine question set
2. Pilot study with 5-10 participants
3. Get IRB approval if needed
4. Deploy on Prolific/MTurk
5. Analyze results
""")


if __name__ == "__main__":
    main()
