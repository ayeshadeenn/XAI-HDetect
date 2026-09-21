import React, { useState, useRef, useCallback } from 'react';
import './App.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const TAXONOMY_COLORS = {
  object:       { bg: '#fff0e6', text: '#c4501a', border: '#f5b894' },
  attribute:    { bg: '#e6f0ff', text: '#1a4fc4', border: '#94b8f5' },
  relationship: { bg: '#e6fff0', text: '#1ac45e', border: '#94f5b8' },
  scene:        { bg: '#f5e6ff', text: '#8a1ac4', border: '#d094f5' },
};

const VERDICT_COLORS = {
  visual_grounding_failure:  { bg: '#fff0e6', text: '#c4501a' },
  language_prior_dominant:   { bg: '#e6f0ff', text: '#1a4fc4' },
  ambiguous:                 { bg: '#f5f5f5', text: '#666' },
};

const VERDICT_LABELS = {
  visual_grounding_failure: 'visual grounding failure',
  language_prior_dominant:  'language prior dominant',
  ambiguous:                'ambiguous',
};

function Badge({ label, colors }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: '20px',
      fontSize: '11px', fontWeight: 600, letterSpacing: '0.03em',
      background: colors.bg, color: colors.text,
      border: `1px solid ${colors.border || colors.bg}`,
      fontFamily: 'Space Mono, monospace',
    }}>
      {label}
    </span>
  );
}

function TokenPill({ token, prob }) {
  const heat = Math.min(1, prob / 0.9);
  const r = Math.round(220 + heat * 35);
  const g = Math.round(220 - heat * 180);
  const b = Math.round(220 - heat * 180);
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      padding: '3px 10px', borderRadius: '4px',
      background: `rgb(${r},${g},${b})`,
      color: heat > 0.5 ? '#fff' : '#333',
      fontFamily: 'Space Mono, monospace', fontSize: '12px', margin: '3px 4px 3px 0',
    }}>
      {token}
      <span style={{ opacity: 0.75, fontSize: 10 }}>{(prob * 100).toFixed(0)}%</span>
    </span>
  );
}

function SpanCard({ span, idx }) {
  const tc = TAXONOMY_COLORS[span.taxonomy] || TAXONOMY_COLORS.object;
  const vc = VERDICT_COLORS[span.causal_verdict] || VERDICT_COLORS.ambiguous;
  const verdictLabel = VERDICT_LABELS[span.causal_verdict] || span.causal_verdict.replace(/_/g, ' ');
  return (
    <div style={{
      background: '#fff', border: '1px solid #e8e8e8',
      borderLeft: `4px solid ${tc.border}`, borderRadius: '8px',
      padding: '14px 16px', marginBottom: 10,
    }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10, marginBottom: 8 }}>
        <span style={{ fontFamily: 'Space Mono, monospace', fontSize: 11, color: '#999', minWidth: 20, paddingTop: 2 }}>{idx}.</span>
        <span style={{ fontFamily: 'DM Sans, sans-serif', fontSize: 15, fontWeight: 500, color: '#1a1a1a', flex: 1 }}>"{span.text}"</span>
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, paddingLeft: 30, marginBottom: 8 }}>
        <Badge label={span.taxonomy} colors={tc} />
        <Badge label={verdictLabel} colors={vc} />
        <span style={{ fontFamily: 'Space Mono, monospace', fontSize: 11, color: '#888', alignSelf: 'center' }}>
          conf: {(span.confidence * 100).toFixed(1)}%
        </span>
        <span style={{ fontFamily: 'Space Mono, monospace', fontSize: 11, color: '#888', alignSelf: 'center' }}>
          grounding: {span.grounding_score.toFixed(3)}
        </span>
      </div>
      {span.explanation && (
        <div style={{
          paddingLeft: 30, fontSize: 12, color: '#555',
          fontFamily: 'DM Sans, sans-serif', lineHeight: 1.6,
          borderTop: '1px solid #f0f0f0', paddingTop: 8, marginTop: 4,
        }}>
          {span.explanation}
        </div>
      )}
    </div>
  );
}

function ExplanationCard({ expl, idx }) {
  const topDrivers = (expl.top_shap || []).slice(0, 3);
  return (
    <div style={{ background: '#fafafa', border: '1px solid #e8e8e8', borderRadius: '8px', padding: '14px 16px', marginBottom: 10 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
        <span style={{ fontFamily: 'Space Mono, monospace', fontSize: 11, color: '#999' }}>{idx}.</span>
        <span style={{ fontFamily: 'Space Mono, monospace', fontSize: 13, fontWeight: 700, color: '#c4501a' }}>"{expl.token.replace('▁', '')}"</span>
        <span style={{ fontFamily: 'Space Mono, monospace', fontSize: 11, color: '#888' }}>p={expl.p_hallucinated.toFixed(3)}</span>
      </div>
      {topDrivers.length > 0 && (
        <div style={{ paddingLeft: 22 }}>
          <div style={{ fontSize: 11, color: '#999', marginBottom: 4, fontFamily: 'DM Sans' }}>Top SHAP drivers:</div>
          {topDrivers.map(([name, val], i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 3 }}>
              <div style={{ width: 80, height: 6, background: '#e8e8e8', borderRadius: 3, overflow: 'hidden' }}>
                <div style={{ width: `${Math.min(100, Math.abs(val) * 30)}%`, height: '100%', background: val > 0 ? '#c4501a' : '#1a4fc4', borderRadius: 3 }} />
              </div>
              <span style={{ fontFamily: 'Space Mono, monospace', fontSize: 10, color: '#666' }}>
                {name.replace(/_/g, ' ')} ({val > 0 ? '+' : ''}{val.toFixed(3)})
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function StatBadge({ label, value }) {
  return (
    <div style={{ textAlign: 'center', padding: '8px 16px', background: '#f5f5f5', borderRadius: 8 }}>
      <div style={{ fontFamily: 'Space Mono, monospace', fontSize: 18, fontWeight: 700, color: '#1a1a1a' }}>{value}</div>
      <div style={{ fontFamily: 'DM Sans, sans-serif', fontSize: 11, color: '#888', marginTop: 2 }}>{label}</div>
    </div>
  );
}

function GlossaryItem({ term, color, description }) {
  return (
    <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
      <span style={{
        fontFamily: 'Space Mono, monospace', fontSize: 10, fontWeight: 700,
        color: color.text, background: color.bg,
        border: `1px solid ${color.border || color.bg}`,
        padding: '2px 8px', borderRadius: 20, whiteSpace: 'nowrap', marginTop: 1,
        minWidth: 'fit-content',
      }}>{term}</span>
      <span style={{ fontSize: 12, color: '#666', lineHeight: 1.5, fontFamily: 'DM Sans, sans-serif' }}>{description}</span>
    </div>
  );
}

export default function App() {
  const [image, setImage]         = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [question, setQuestion]   = useState('Describe the image in detail.');
  const [result, setResult]       = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);
  const [dragOver, setDragOver]   = useState(false);
  const fileRef = useRef();

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) return;
    setImageFile(file);
    setImage(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const handleAnalyze = async () => {
    if (!imageFile) return;

    const q = question.trim();
    if (q.length < 5) {
      setError('Please enter a meaningful text prompt about the image (at least 5 characters).');
      return;
    }
    if (/^[^a-zA-Z]+$/.test(q)) {
      setError('Please enter a meaningful question containing words.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const form = new FormData();
    form.append('image', imageFile);
    form.append('question', question);

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 180000);

    try {
      const res = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        body: form,
        signal: controller.signal,
      });
      clearTimeout(timeout);
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Request failed');
      }
      setResult(await res.json());
    } catch (e) {
      clearTimeout(timeout);
      if (e.name === 'AbortError') {
        setError('Request timed out. The server may be starting up — please try again in 30 seconds.');
      } else {
        setError(e.message);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: '100vh', background: '#f8f7f4', fontFamily: 'DM Sans, sans-serif' }}>

      {/* Header */}
      <header style={{ background: '#0f0f0f', padding: '0 32px', height: 60, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ width: 28, height: 28, background: '#c4501a', borderRadius: 6, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <span style={{ color: '#fff', fontFamily: 'Space Mono', fontSize: 12, fontWeight: 700 }}>X</span>
          </div>
          <span style={{ color: '#fff', fontFamily: 'Space Mono, monospace', fontSize: 14, fontWeight: 700, letterSpacing: '-0.02em' }}>XAI-HDetect</span>
        </div>
        <span style={{ fontFamily: 'Space Mono, monospace', fontSize: 11, color: '#555' }}>Explainable Hallucination Detection for VLMs</span>
      </header>

      <main style={{ maxWidth: 1100, margin: '0 auto', padding: '32px 24px' }}>

        {/* Hero section */}
        <div style={{ marginBottom: 32, textAlign: 'center' }}>
          <h1 style={{ fontFamily: 'Space Mono, monospace', fontSize: 28, fontWeight: 700, color: '#0f0f0f', margin: '0 0 12px', letterSpacing: '-0.03em' }}>
            Token-Level Hallucination Detection
          </h1>
          <p style={{ color: '#666', fontSize: 14, maxWidth: 580, margin: '0 auto 16px', lineHeight: 1.7 }}>
            Vision-language models hallucinate. They describe things that aren't there, with complete confidence.
            XAI-HDetect pinpoints exactly which words are fabricated, what type of error they represent,
            and why the model generated them.
          </p>

        </div>

        {/* Input area */}
        <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 20, marginBottom: 24 }}>

          {/* Image upload — larger */}
          <div
            onDragOver={e => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileRef.current.click()}
            style={{
              border: `2px dashed ${dragOver ? '#c4501a' : '#d0cfc9'}`,
              borderRadius: 12, background: dragOver ? '#fff5f0' : '#fff',
              minHeight: 340, display: 'flex', alignItems: 'center', justifyContent: 'center',
              cursor: 'pointer', overflow: 'hidden', transition: 'all 0.2s',
            }}
          >
            <input ref={fileRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={e => handleFile(e.target.files[0])} />
            {image ? (
              <img src={image} alt="uploaded" style={{ width: '100%', height: '100%', objectFit: 'contain', maxHeight: 380 }} />
            ) : (
              <div style={{ textAlign: 'center', padding: 32 }}>
                <div style={{ fontSize: 40, marginBottom: 12 }}>🖼</div>
                <div style={{ color: '#888', fontSize: 13, marginBottom: 6 }}>Drop image here or click to upload</div>
                <div style={{ color: '#bbb', fontSize: 11, fontFamily: 'Space Mono, monospace' }}>JPG · PNG · WebP supported</div>
              </div>
            )}
          </div>

          {/* Controls */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div>
              <label style={{ display: 'block', fontFamily: 'Space Mono, monospace', fontSize: 11, color: '#888', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                Question / Prompt
              </label>
              <textarea
                value={question}
                onChange={e => setQuestion(e.target.value)}
                rows={5}
                placeholder="e.g. Describe the image in detail."
                style={{ width: '100%', padding: '12px 14px', borderRadius: 8, border: '1px solid #e0dfd9', background: '#fff', fontFamily: 'DM Sans, sans-serif', fontSize: 14, color: '#1a1a1a', resize: 'vertical', outline: 'none', boxSizing: 'border-box', lineHeight: 1.6 }}
              />
              <div style={{ fontSize: 11, color: '#bbb', marginTop: 6, fontFamily: 'DM Sans' }}>
                Ask the model to describe the image. The system detects hallucinated content in the generated response.
              </div>
            </div>

            <button
              onClick={handleAnalyze}
              disabled={!imageFile || loading}
              style={{ padding: '14px 24px', background: imageFile && !loading ? '#c4501a' : '#ddd', color: imageFile && !loading ? '#fff' : '#aaa', border: 'none', borderRadius: 8, fontFamily: 'Space Mono, monospace', fontSize: 13, fontWeight: 700, cursor: imageFile && !loading ? 'pointer' : 'not-allowed', letterSpacing: '0.02em', transition: 'all 0.2s' }}>
              {loading ? 'Analysing...' : 'Detect Hallucinations'}
            </button>

            {loading && (
              <div style={{ padding: '12px 16px', background: '#fff5f0', borderRadius: 8, border: '1px solid #f5b894', fontSize: 12, color: '#c4501a', fontFamily: 'Space Mono, monospace' }}>
                Running pipeline… (~20–30s on warm server, up to 2 min on first load)
              </div>
            )}

            {error && (
              <div style={{ padding: '12px 16px', background: '#fff0f0', borderRadius: 8, border: '1px solid #f5a0a0', fontSize: 13, color: '#c41a1a', fontFamily: 'DM Sans' }}>
                {error}
              </div>
            )}

            {/* Quick guide */}
            {!result && !loading && (
              <div style={{ padding: '14px 16px', background: '#fff', border: '1px solid #e8e8e8', borderRadius: 8 }}>
                <div style={{ fontFamily: 'Space Mono, monospace', fontSize: 10, color: '#999', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 10 }}>How it works</div>
                {[
                  '1. Upload any image',
                  '2. Enter a descriptive question',
                  '3. The system generates a caption using LLaVA',
                  '4. Hallucinated tokens are identified and explained',
                ].map((step, i) => (
                  <div key={i} style={{ fontSize: 12, color: '#666', fontFamily: 'DM Sans', marginBottom: 5, lineHeight: 1.5 }}>{step}</div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Results */}
        {result && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

            {/* Stats */}
            <div style={{ background: '#fff', border: '1px solid #e8e8e8', borderRadius: 12, padding: '16px 24px', display: 'flex', gap: 16 }}>
              <StatBadge label="Tokens detected" value={result.detected_tokens?.length || 0} />
              <StatBadge label="Spans detected" value={result.spans?.length || 0} />
            </div>

            {/* Caption */}
            <div style={{ background: '#fff', border: '1px solid #e8e8e8', borderRadius: 12, padding: '20px 24px' }}>
              <div style={{ fontFamily: 'Space Mono, monospace', fontSize: 11, color: '#999', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 }}>Generated Caption</div>
              <div style={{ fontSize: 11, color: '#bbb', marginBottom: 10, fontFamily: 'DM Sans' }}>LLaVA-1.5-7B response — hallucination detection is applied to this output</div>
              <p style={{ fontSize: 15, color: '#1a1a1a', lineHeight: 1.7, margin: 0 }}>{result.caption}</p>
            </div>

            {/* Tokens + Spans */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
              <div style={{ background: '#fff', border: '1px solid #e8e8e8', borderRadius: 12, padding: '20px 24px' }}>
                <div style={{ fontFamily: 'Space Mono, monospace', fontSize: 11, color: '#999', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 4 }}>Hallucinated Tokens</div>
                <div style={{ fontSize: 11, color: '#bbb', marginBottom: 12, fontFamily: 'DM Sans' }}>LightGBM detector — calibrated probabilities. Darker red indicates higher hallucination confidence.</div>
                {result.detected_tokens?.length === 0
                  ? <p style={{ color: '#888', fontSize: 13 }}>No hallucinated tokens detected.</p>
                  : <div>{result.detected_tokens?.map((t, i) => <TokenPill key={i} token={t.token} prob={t.prob} />)}</div>
                }
              </div>

              <div style={{ background: '#fff', border: '1px solid #e8e8e8', borderRadius: 12, padding: '20px 24px' }}>
                <div style={{ fontFamily: 'Space Mono, monospace', fontSize: 11, color: '#999', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 4 }}>Hallucinated Spans</div>
                <div style={{ fontSize: 11, color: '#bbb', marginBottom: 12, fontFamily: 'DM Sans' }}>Consecutive hallucinated tokens grouped into phrases, classified by type and explained.</div>
                {result.spans?.length === 0
                  ? <p style={{ color: '#888', fontSize: 13 }}>No hallucinated spans detected.</p>
                  : result.spans?.map((sp, i) => <SpanCard key={i} span={sp} idx={i + 1} />)
                }
              </div>
            </div>

            {/* Grad-CAM + SHAP */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
              <div style={{ background: '#fff', border: '1px solid #e8e8e8', borderRadius: 12, padding: '20px 24px' }}>
                <div style={{ fontFamily: 'Space Mono, monospace', fontSize: 11, color: '#999', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 4 }}>Grad-CAM — VLM Visual Grounding</div>
                <div style={{ fontSize: 11, color: '#bbb', marginBottom: 12, fontFamily: 'DM Sans' }}>Which image regions the VLM attended to when generating the hallucinated token.</div>
                {result.gradcam_image ? (
                  <div>
                    <img src={`data:image/png;base64,${result.gradcam_image}`} alt="Grad-CAM" style={{ width: '100%', borderRadius: 8, marginBottom: 10 }} />
                    <div style={{ fontFamily: 'Space Mono, monospace', fontSize: 11, color: '#c4501a', background: '#fff5f0', padding: '8px 12px', borderRadius: 6 }}>
                      Token: "{result.gradcam_token}" | p={result.gradcam_prob}
                    </div>
                  </div>
                ) : (
                  <div style={{ padding: '24px', background: '#f5f5f5', borderRadius: 8, textAlign: 'center', color: '#888', fontSize: 13 }}>No high-confidence hallucinated tokens for Grad-CAM</div>
                )}
              </div>

              <div style={{ background: '#fff', border: '1px solid #e8e8e8', borderRadius: 12, padding: '20px 24px' }}>
                <div style={{ fontFamily: 'Space Mono, monospace', fontSize: 11, color: '#999', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 4 }}>Token-Level Feature Attribution</div>
                <div style={{ fontSize: 11, color: '#bbb', marginBottom: 12, fontFamily: 'DM Sans' }}>SHAP drivers per flagged token. Red bars increase hallucination confidence; blue bars decrease it.</div>
                {result.explanations?.length === 0
                  ? <p style={{ color: '#888', fontSize: 13 }}>No explanations available.</p>
                  : result.explanations?.slice(0, 5).map((ex, i) => <ExplanationCard key={i} expl={ex} idx={i + 1} />)
                }
              </div>
            </div>

            {/* Explanation framework */}
            <div style={{ background: '#fff', border: '1px solid #e8e8e8', borderRadius: 12, padding: '20px 24px' }}>
              <div style={{ fontFamily: 'Space Mono, monospace', fontSize: 11, color: '#999', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 20 }}>Explanation Framework</div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 32 }}>

                {/* Explanation methods */}
                <div>
                  <div style={{ fontFamily: 'Space Mono, monospace', fontSize: 10, color: '#ccc', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 14 }}>Explanation methods</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                    {[
                      { label: 'SHAP', color: { bg: '#fff0e6', text: '#c4501a', border: '#f5b894' }, desc: 'Detector-side explanation. Shows which features drove the classifier\'s decision for each flagged token.' },
                      { label: 'Grad-CAM', color: { bg: '#e6f0ff', text: '#1a4fc4', border: '#94b8f5' }, desc: 'VLM-side explanation. Highlights which image regions the vision encoder attended to when generating the hallucinated token.' },
                    ].map((item, i) => (
                      <GlossaryItem key={i} term={item.label} color={item.color} description={item.desc} />
                    ))}
                  </div>
                </div>

                {/* Score definitions */}
                <div>
                  <div style={{ fontFamily: 'Space Mono, monospace', fontSize: 10, color: '#ccc', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 14 }}>Score definitions</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                    {[
                      { label: 'conf %', color: { bg: '#f5f5f5', text: '#555' }, desc: 'Mean calibrated hallucination probability across all tokens in the span.' },
                      { label: 'grounding score', color: { bg: '#f5f5f5', text: '#555' }, desc: 'Mean CLIP regional similarity score. Lower values indicate poor visual support for the span.' },
                      { label: 'p = value', color: { bg: '#f5f5f5', text: '#555' }, desc: 'Per-token calibrated probability in the SHAP panel. Detector confidence that this token is hallucinated.' },
                    ].map((item, i) => (
                      <GlossaryItem key={i} term={item.label} color={item.color} description={item.desc} />
                    ))}
                  </div>
                </div>

                {/* Span taxonomy */}
                <div>
                  <div style={{ fontFamily: 'Space Mono, monospace', fontSize: 10, color: '#ccc', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 14 }}>Hallucination types</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                    {[
                      { label: 'object', color: TAXONOMY_COLORS.object, desc: 'A non-existent object is described.' },
                      { label: 'attribute', color: TAXONOMY_COLORS.attribute, desc: 'A wrong property (colour, size, shape) is assigned to a real object.' },
                      { label: 'relationship', color: TAXONOMY_COLORS.relationship, desc: 'An incorrect spatial or action relationship between objects is stated.' },
                      { label: 'scene', color: TAXONOMY_COLORS.scene, desc: 'The overall scene or setting is incorrectly described.' },
                    ].map((item, i) => (
                      <GlossaryItem key={i} term={item.label} color={item.color} description={item.desc} />
                    ))}
                  </div>
                </div>

                {/* Verdicts */}
                <div>
                  <div style={{ fontFamily: 'Space Mono, monospace', fontSize: 10, color: '#ccc', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 14 }}>Visual dependence verdicts</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                    {[
                      { label: 'visual grounding failure', color: VERDICT_COLORS.visual_grounding_failure, desc: 'The model was misled by the image - hallucination is driven by visual content.' },
                      { label: 'language prior dominant', color: VERDICT_COLORS.language_prior_dominant, desc: 'The model would have generated this token even without the image - driven by language statistics.' },
                      { label: 'ambiguous', color: VERDICT_COLORS.ambiguous, desc: 'Signal is weak - hallucination cause cannot be reliably determined.' },
                    ].map((item, i) => (
                      <GlossaryItem key={i} term={item.label} color={item.color} description={item.desc} />
                    ))}
                  </div>
                </div>

              </div>
            </div>

          </div>
        )}
      </main>
    </div>
  );
}
