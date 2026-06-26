from flask import Flask, request, jsonify, render_template_string
import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast, BertForQuestionAnswering

# ── Config ─────────────────────────────────────────────────
MODEL_DIR  = "outputs/best_model"
MAX_LENGTH = 384
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load Model ─────────────────────────────────────────────
print(f"[INFO] Loading model from: {MODEL_DIR}")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
model     = BertForQuestionAnswering.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()
print("[INFO] Model ready!")

# ── Flask App ──────────────────────────────────────────────
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>QA Model Tester</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; padding: 30px 20px; }
    .container { max-width: 800px; margin: 0 auto; }
    h1 { text-align: center; font-size: 2rem; margin-bottom: 8px; color: #38bdf8; }
    .subtitle { text-align: center; color: #94a3b8; margin-bottom: 30px; font-size: 0.95rem; }
    .card { background: #1e293b; border-radius: 12px; padding: 24px; margin-bottom: 20px; border: 1px solid #334155; }
    label { display: block; font-weight: 600; margin-bottom: 8px; color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }
    textarea, input { width: 100%; background: #0f172a; border: 1px solid #334155; border-radius: 8px; color: #e2e8f0; padding: 12px; font-size: 0.95rem; font-family: inherit; resize: vertical; transition: border 0.2s; }
    textarea:focus, input:focus { outline: none; border-color: #38bdf8; }
    textarea { min-height: 140px; }
    input { min-height: 48px; }
    button { width: 100%; padding: 14px; background: #0ea5e9; color: white; border: none; border-radius: 8px; font-size: 1rem; font-weight: 600; cursor: pointer; transition: background 0.2s; margin-top: 4px; }
    button:hover { background: #0284c7; }
    button:disabled { background: #334155; cursor: not-allowed; }
    .result { display: none; background: #1e293b; border-radius: 12px; padding: 24px; border: 1px solid #334155; }
    .answer-box { background: #0f172a; border-radius: 8px; padding: 16px; margin: 12px 0; border-left: 4px solid #38bdf8; }
    .answer-text { font-size: 1.3rem; font-weight: 700; color: #38bdf8; }
    .badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; margin-top: 8px; }
    .high   { background: #14532d; color: #4ade80; }
    .medium { background: #713f12; color: #fbbf24; }
    .low    { background: #7f1d1d; color: #f87171; }
    .score  { color: #64748b; font-size: 0.85rem; margin-top: 6px; }
    .loading { text-align: center; color: #38bdf8; padding: 20px; display: none; }
    .examples { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
    .ex-btn { width: auto; padding: 6px 14px; font-size: 0.8rem; background: #1e3a5f; border: 1px solid #38bdf8; color: #38bdf8; border-radius: 20px; cursor: pointer; transition: all 0.2s; margin-top: 0; }
    .ex-btn:hover { background: #38bdf8; color: #0f172a; }
  </style>
</head>
<body>
  <div class="container">
    <h1>🤖 QA Model Tester</h1>
    <p class="subtitle">Fine-Tuned BERT on SQuAD — Test your model</p>

    <div class="card">
      <label>📄 Context (النص)</label>
      <textarea id="context" placeholder="Paste your paragraph here..."></textarea>
      <div class="examples">
        <span style="color:#64748b;font-size:0.8rem;align-self:center;">Examples:</span>
        <button class="ex-btn" onclick="loadExample('ai')">AI History</button>
        <button class="ex-btn" onclick="loadExample('bert')">BERT</button>
        <button class="ex-btn" onclick="loadExample('squad')">SQuAD</button>
      </div>
    </div>

    <div class="card">
      <label>❓ Question (السؤال)</label>
      <input type="text" id="question" placeholder="Ask something about the context..." onkeydown="if(event.key==='Enter') askQuestion()">
      <button onclick="askQuestion()" id="btn">Ask ➤</button>
    </div>

    <div class="loading" id="loading">⏳ Thinking...</div>

    <div class="result" id="result">
      <label>✅ Answer</label>
      <div class="answer-box">
        <div class="answer-text" id="answer-text"></div>
        <div id="badge-div"></div>
        <div class="score" id="score-text"></div>
      </div>
    </div>
  </div>

  <script>
    const EXAMPLES = {
      ai: {
        ctx: `Artificial intelligence (AI) is intelligence demonstrated by machines. Alan Turing proposed the Turing test in 1950. Deep Blue, developed by IBM, defeated chess world champion Garry Kasparov in 1997. In 2016, AlphaGo, developed by Google DeepMind, defeated Go world champion Lee Sedol.`,
        q: "When did Alan Turing propose the Turing Test?"
      },
      bert: {
        ctx: `BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based machine learning technique developed by Google. BERT was created and published in 2018 by Jacob Devlin and his colleagues from Google. BERT-base has 12 transformer layers and 110 million parameters.`,
        q: "Who created BERT?"
      },
      squad: {
        ctx: `The Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset consisting of questions posed by crowdworkers on Wikipedia articles. SQuAD 1.1 contains 100,000+ question-answer pairs on 500+ articles. The two main evaluation metrics are Exact Match (EM) and F1 Score.`,
        q: "How many questions does SQuAD 1.1 contain?"
      }
    };

    function loadExample(key) {
      document.getElementById('context').value  = EXAMPLES[key].ctx;
      document.getElementById('question').value = EXAMPLES[key].q;
    }

    async function askQuestion() {
      const context  = document.getElementById('context').value.trim();
      const question = document.getElementById('question').value.trim();
      if (!context || !question) { alert('Please fill in both fields!'); return; }

      document.getElementById('btn').disabled    = true;
      document.getElementById('loading').style.display = 'block';
      document.getElementById('result').style.display  = 'none';

      try {
        const res  = await fetch('/predict', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({context, question})
        });
        const data = await res.json();

        document.getElementById('answer-text').textContent = data.answer || '(no answer found)';
        document.getElementById('score-text').textContent  = `Score: ${data.score.toFixed(4)}`;

        const score = data.score;
        let badge = '';
        if      (score > 1.5) badge = '<span class="badge high">🟢 High Confidence</span>';
        else if (score > 1.0) badge = '<span class="badge medium">🟡 Medium Confidence</span>';
        else                  badge = '<span class="badge low">🔴 Low Confidence</span>';
        document.getElementById('badge-div').innerHTML = badge;

        document.getElementById('result').style.display = 'block';
      } catch(e) {
        alert('Error: ' + e.message);
      } finally {
        document.getElementById('btn').disabled    = false;
        document.getElementById('loading').style.display = 'none';
      }
    }
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    data     = request.json
    question = data.get("question", "")
    context  = data.get("context", "")

    inputs = tokenizer(
        question, context,
        max_length=MAX_LENGTH,
        truncation="only_second",
        return_offsets_mapping=True,
        return_tensors="pt",
        padding="max_length",
    )
    offset_mapping = inputs.pop("offset_mapping")[0].tolist()
    seq_ids        = inputs.sequence_ids(0)
    inputs         = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0].cpu()
    end_logits   = outputs.end_logits[0].cpu()

    for idx, sid in enumerate(seq_ids):
        if sid != 1:
            start_logits[idx] = -1e9
            end_logits[idx]   = -1e9

    start_probs = F.softmax(start_logits, dim=-1).numpy()
    end_probs   = F.softmax(end_logits,   dim=-1).numpy()

    best_score, best_start, best_end = -1e9, 0, 0
    for s in range(len(start_probs)):
        for e in range(s, min(s + 50, len(end_probs))):
            score = start_probs[s] + end_probs[e]
            if score > best_score:
                best_score, best_start, best_end = score, s, e

    char_start = offset_mapping[best_start][0] if offset_mapping[best_start] else 0
    char_end   = offset_mapping[best_end][1]   if offset_mapping[best_end]   else 0
    answer     = context[char_start:char_end]

    return jsonify({"answer": answer, "score": float(best_score)})

if __name__ == "__main__":
    print("\n[INFO] Open your browser at: http://localhost:5000\n")
    app.run(debug=False, port=5000)