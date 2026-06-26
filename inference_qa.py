
import os
import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering

MODEL_DIR  = "deepset/bert-base-uncased-squad2"
MAX_LENGTH = 384
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_dir: str):
    print(f"[INFO] Loading model from: {model_dir}")
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model     = BertForQuestionAnswering.from_pretrained(model_dir)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def answer_question(question: str, context: str,
                    tokenizer, model,
                    max_answer_length: int = 50) -> dict:
    inputs = tokenizer(
        question,
        context,
        max_length=MAX_LENGTH,
        truncation="only_second",
        return_offsets_mapping=True,
        return_tensors="pt",
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")[0].tolist()
    seq_ids        = inputs.sequence_ids(0)

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0].cpu()
    end_logits   = outputs.end_logits[0].cpu()

    for idx, sid in enumerate(seq_ids):
        if sid != 1:
            start_logits[idx] = -1e9
            end_logits[idx]   = -1e9

    import torch.nn.functional as F
    start_probs = F.softmax(start_logits, dim=-1).numpy()
    end_probs   = F.softmax(end_logits,   dim=-1).numpy()

    import numpy as np
    best_score  = -1e9
    best_start  = 0
    best_end    = 0
    for s in range(len(start_probs)):
        for e in range(s, min(s + max_answer_length, len(end_probs))):
            score = start_probs[s] + end_probs[e]
            if score > best_score:
                best_score = score
                best_start = s
                best_end   = e

    char_start = offset_mapping[best_start][0] if offset_mapping[best_start] else 0
    char_end   = offset_mapping[best_end][1]   if offset_mapping[best_end]   else 0
    answer     = context[char_start:char_end]

    return {
        "answer":     answer,
        "score":      float(best_score),
        "char_start": char_start,
        "char_end":   char_end,
    }


DEMO_CONTEXTS = {
    "AI History": """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural 
intelligence displayed by animals including humans. AI research has been defined as the field of 
study of intelligent agents, which refers to any system that perceives its environment and takes 
actions that maximize its chance of achieving its goals. The term "artificial intelligence" had 
previously been used to describe machines that mimic and display human cognitive skills associated 
with the human mind, such as learning and problem-solving. This definition has since been rejected 
by major AI researchers who now describe AI in terms of rationality and acting rationally. 
Alan Turing proposed the Turing test in 1950. Deep Blue, developed by IBM, defeated chess 
world champion Garry Kasparov in 1997. In 2016, AlphaGo, developed by Google DeepMind, 
defeated Go world champion Lee Sedol.
""",
    "BERT": """
BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based machine 
learning technique for natural language processing (NLP) pre-training developed by Google. 
BERT was created and published in 2018 by Jacob Devlin and his colleagues from Google. 
Google has adopted BERT in its search engine since late 2019. BERT is pre-trained on two 
unsupervised tasks: Masked Language Modeling and Next Sentence Prediction. The original 
BERT paper demonstrated state-of-the-art results on eleven NLP tasks including Question 
Answering (SQuAD v1.1 and v2.0), Natural Language Inference (MNLI), and others. 
BERT-base has 12 transformer layers (blocks), 12 attention heads, and 110 million parameters.
BERT-large has 24 transformer layers, 16 attention heads, and 340 million parameters.
""",
    "SQuAD Dataset": """
The Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset consisting 
of questions posed by crowdworkers on a set of Wikipedia articles. The answer to every question 
is a segment of text, or span, from the corresponding reading passage. SQuAD 1.1 contains 
100,000+ question-answer pairs on 500+ articles. SQuAD 2.0 combines the 100,000 questions in 
SQuAD 1.1 with over 50,000 unanswerable questions written adversarially by crowdworkers to 
look similar to answerable ones. SQuAD was introduced by Rajpurkar et al. in 2016. 
The two main evaluation metrics for SQuAD are Exact Match (EM) and F1 Score.
""",
}

DEMO_QUESTIONS = {
    "AI History": [
        "When did Alan Turing propose the Turing Test?",
        "Who developed Deep Blue?",
        "When did AlphaGo defeat Lee Sedol?",
    ],
    "BERT": [
        "Who created BERT?",
        "How many parameters does BERT-base have?",
        "What are the two pre-training tasks used by BERT?",
    ],
    "SQuAD Dataset": [
        "What does SQuAD stand for?",
        "How many questions does SQuAD 1.1 contain?",
        "What are the evaluation metrics for SQuAD?",
    ],
}


def run_demo(tokenizer, model):
    print("\n" + "="*60)
    print("  QA DEMO — Fine-Tuned BERT on SQuAD")
    print("="*60)

    for topic, context in DEMO_CONTEXTS.items():
        print(f"\n📖 Topic: {topic}")
        print("-"*50)
        for question in DEMO_QUESTIONS[topic]:
            result = answer_question(question, context, tokenizer, model)
            print(f"  Q: {question}")
            print(f"  A: {result['answer']!r}  (score: {result['score']:.3f})")
        print()


def interactive_mode(tokenizer, model):
    print("\n" + "="*60)
    print("  INTERACTIVE QA MODE  (type 'quit' to exit)")
    print("="*60)

    context = ""
    while True:
        print("\nOptions: [1] New context  [2] Ask question  [3] Quit")
        choice = input("Choice: ").strip()

        if choice == "1":
            print("Paste context (press Enter twice when done):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            context = " ".join(lines)
            print(f"[Context set: {len(context)} chars]")

        elif choice == "2":
            if not context:
                print("[!] Please set a context first.")
                continue
            question = input("Question: ").strip()
            if question.lower() == "quit":
                break
            result = answer_question(question, context, tokenizer, model)
            print(f"\nAnswer : {result['answer']!r}")
            print(f"Score  : {result['score']:.4f}")

            # تفسير الـ Score للمستخدم
            score = result['score']
            if score > 1.5:
                confidence = "🟢 High confidence"
            elif score > 1.0:
                confidence = "🟡 Medium confidence"
            else:
                confidence = "🔴 Low confidence — try rephrasing the question"
            print(f"Status : {confidence}")
            print(f"Chars  : [{result['char_start']}, {result['char_end']}]")

        elif choice == "3" or choice.lower() == "quit":
            break


if __name__ == "__main__":
    import sys

    model_dir = MODEL_DIR
    tokenizer, model = load_model(model_dir)

    if "--interactive" in sys.argv:
        interactive_mode(tokenizer, model)
    else:
        run_demo(tokenizer, model)