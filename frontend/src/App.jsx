import { useEffect, useState } from "react";
import "./App.css";

// Where the backend lives. When you deploy, change this to your Render URL.
const API_URL = "http://127.0.0.1:8000";

// Realistic samples spanning different document types, so you (and your demo
// audience) can try it instantly. The "Try a sample" button cycles through them.
const SAMPLE_TEXTS = [
  "The patient presented with acute hypertension and suspected myocardial " +
    "infarction. An echocardiogram revealed myocardial ischemia. We initiated " +
    "anticoagulant therapy and scheduled a biopsy to rule out malignant neoplasm.",

  "At discharge the patient remained hemodynamically stable following " +
    "cholecystectomy. Continue prophylactic antibiotics and analgesics as " +
    "needed. Return immediately if you develop dyspnea, febrile symptoms, or " +
    "signs of postoperative hemorrhage.",

  "The radiograph demonstrates bilateral pulmonary infiltrates consistent with " +
    "pneumonia. There is no evidence of pneumothorax or pleural effusion. Mild " +
    "cardiomegaly is noted. Correlation with the clinical presentation is " +
    "recommended.",

  "Laboratory results indicate hyperglycemia and an elevated hemoglobin A1c, " +
    "consistent with poorly controlled diabetes mellitus. The lipid panel shows " +
    "hypercholesterolemia. We recommend initiating metformin and dietary " +
    "modification to reduce cardiovascular risk.",

  "The patient experienced a transient ischemic attack with acute aphasia and " +
    "unilateral paresthesia. Neuroimaging excluded an intracranial hemorrhage. " +
    "Antiplatelet therapy was started to prevent a subsequent cerebrovascular " +
    "accident.",
];

export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [tip, setTip] = useState(null);
  const [sampleIndex, setSampleIndex] = useState(0);

  function loadSample() {
    setText(SAMPLE_TEXTS[sampleIndex]);
    setSampleIndex((i) => (i + 1) % SAMPLE_TEXTS.length);
  }

  // The tooltip is positioned with fixed coordinates relative to the word.
  // It must be dismissed on scroll/resize too: the browser won't fire
  // mouseleave when the page scrolls out from under a stationary cursor,
  // which is what left the old CSS-hover tooltip stuck on screen.
  useEffect(() => {
    if (!tip) return;
    const dismiss = () => setTip(null);
    window.addEventListener("scroll", dismiss, true);
    window.addEventListener("resize", dismiss);
    return () => {
      window.removeEventListener("scroll", dismiss, true);
      window.removeEventListener("resize", dismiss);
    };
  }, [tip]);

  function showTip(e, definition) {
    const r = e.currentTarget.getBoundingClientRect();
    const above = r.top > 90;
    const margin = 12;
    const left = Math.min(
      Math.max(r.left + r.width / 2, margin + 140),
      window.innerWidth - margin - 140
    );
    setTip({
      text: definition,
      left,
      top: above ? r.top - 8 : r.bottom + 8,
      placement: above ? "above" : "below",
    });
  }

  async function analyze() {
    setError("");
    setResult(null);
    if (!text.trim()) {
      setError("Please paste or type some text first.");
      return;
    }
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(
        "Couldn't reach the backend. Make sure the server is running " +
          "(uvicorn app:app --reload). Details: " + err.message
      );
    } finally {
      setLoading(false);
    }
  }

  // Build the list of unique flagged terms that have a definition,
  // for the glossary panel below the text.
  const definedTerms = result
    ? Array.from(
        new Map(
          result.tokens
            .filter((t) => t.complex && t.definition)
            .map((t) => [t.text.toLowerCase(), t])
        ).values()
      )
    : [];

  return (
    <div className="page">
      <header className="hero">
        <div className="logo">✚</div>
        <h1>ClaruRx</h1>
        <p className="tagline">
          Spotting the words that stand between patients and understanding.
        </p>
      </header>

      <main className="workspace">
        <section className="input-card">
          <label htmlFor="input">Paste a medical document, lab result, or discharge note</label>
          <textarea
            id="input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste text here..."
            rows={6}
          />
          <div className="button-row">
            <button className="primary" onClick={analyze} disabled={loading}>
              {loading ? "Analyzing..." : "Analyze readability"}
            </button>
            <button className="ghost" onClick={loadSample}>
              Try a sample
            </button>
          </div>
          {error && <p className="error">{error}</p>}
        </section>

        {result && (
          <section className="results">
            <div className="stats-row">
              <div className="stat">
                <span className="stat-number">{result.readability.gradeLevel}</span>
                <span className="stat-label">reading grade level</span>
              </div>
              <div className="stat">
                <span className="stat-number">{result.stats.complexWords}</span>
                <span className="stat-label">complex words found</span>
              </div>
              <div className="stat">
                <span className="stat-number">{result.stats.percentComplex}%</span>
                <span className="stat-label">of the text</span>
              </div>
            </div>

            <div className="annotated">
              {result.tokens.map((token, i) =>
                token.complex ? (
                  <span
                    key={i}
                    className="flagged"
                    onMouseEnter={(e) =>
                      showTip(e, token.definition || "Likely hard to understand")
                    }
                    onMouseLeave={() => setTip(null)}
                  >
                    {token.text}
                  </span>
                ) : (
                  <span key={i}>{token.text}</span>
                )
              )}
            </div>

            {definedTerms.length > 0 && (
              <div className="glossary">
                <h2>Plain-language meanings</h2>
                <ul>
                  {definedTerms.map((t, i) => (
                    <li key={i}>
                      <strong>{t.text}</strong>
                      <span>{t.definition}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </section>
        )}
      </main>

      <footer className="disclaimer">
        ClaruRx is a reading aid, not medical advice. It does not replace a
        doctor, nurse, or pharmacist. Always confirm health decisions with a
        qualified professional.
      </footer>

      {tip && (
        <div
          className={`tooltip ${tip.placement}`}
          style={{ left: tip.left, top: tip.top }}
        >
          {tip.text}
        </div>
      )}
    </div>
  );
}
