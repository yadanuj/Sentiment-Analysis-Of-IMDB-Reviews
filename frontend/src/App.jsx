import { useState } from 'react';
import './index.css';

function App() {
  const [review, setReview] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const analyzeSentiment = async () => {
    if (!review.trim()) return;
    
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await fetch('http://localhost:5001/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ review })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to analyze sentiment');
      }

      setPrediction(data.prediction === 1 ? 'POSITIVE' : 'NEGATIVE');
      setConfidence(data.confidence);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="blob blob-1"></div>
      <div className="blob blob-2"></div>
      <div className="blob blob-3"></div>

      <div className="glass-panel" style={{ padding: '3rem', position: 'relative', overflow: 'hidden' }}>
        <div style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
          <h1 style={{ fontSize: '3rem', fontWeight: 800, background: 'linear-gradient(to right, #a855f7, #ec4899)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', marginBottom: '0.5rem' }}>
            Neural Sentiment
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '1.1rem' }}>
            Powered by Advanced N-Gram ML Models
          </p>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          
          {/* Quick Fill Options */}
          <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
            <button
              onClick={() => setReview("The movie was undeniably a masterpiece. From the flawless cinematography to the brilliant acting, every single scene was captivating. Best film of the year by far!")}
              style={{
                flex: 1,
                padding: '0.8rem 1rem',
                borderRadius: '12px',
                border: '1px solid rgba(16, 185, 129, 0.3)',
                background: 'rgba(16, 185, 129, 0.1)',
                color: 'var(--success)',
                cursor: 'pointer',
                fontSize: '0.95rem',
                transition: 'all 0.2s',
                fontFamily: 'inherit'
              }}
              onMouseOver={(e) => { e.target.style.background = 'rgba(16, 185, 129, 0.2)'; e.target.style.transform = 'translateY(-2px)' }}
              onMouseOut={(e) => { e.target.style.background = 'rgba(16, 185, 129, 0.1)'; e.target.style.transform = 'translateY(0)' }}
            >
              ✨ Insert Positive Sample
            </button>

            <button
              onClick={() => setReview("This film was an absolute disaster. The acting was hollow, the plot made absolutely no sense, and I couldn't wait for it to end. Do not waste your time or money.")}
              style={{
                flex: 1,
                padding: '0.8rem 1rem',
                borderRadius: '12px',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                background: 'rgba(239, 68, 68, 0.1)',
                color: 'var(--danger)',
                cursor: 'pointer',
                fontSize: '0.95rem',
                transition: 'all 0.2s',
                fontFamily: 'inherit'
              }}
              onMouseOver={(e) => { e.target.style.background = 'rgba(239, 68, 68, 0.2)'; e.target.style.transform = 'translateY(-2px)' }}
              onMouseOut={(e) => { e.target.style.background = 'rgba(239, 68, 68, 0.1)'; e.target.style.transform = 'translateY(0)' }}
            >
              📉 Insert Negative Sample
            </button>
          </div>

          <textarea
            value={review}
            onChange={(e) => setReview(e.target.value)}
            placeholder="Type a movie review here (e.g. 'The movie was good but the acting was completely terrible...')"
            style={{
              width: '100%',
              minHeight: '150px',
              padding: '1.5rem',
              borderRadius: '16px',
              border: '2px solid rgba(255, 255, 255, 0.1)',
              background: 'rgba(0, 0, 0, 0.2)',
              color: 'var(--text-primary)',
              fontSize: '1.1rem',
              fontFamily: 'inherit',
              resize: 'vertical',
              outline: 'none',
              transition: 'all 0.3s ease',
            }}
            onFocus={(e) => e.target.style.borderColor = 'var(--accent-color)'}
            onBlur={(e) => e.target.style.borderColor = 'rgba(255, 255, 255, 0.1)'}
          />

          <button
            onClick={analyzeSentiment}
            disabled={loading || !review.trim()}
            style={{
              padding: '1rem 2rem',
              borderRadius: '12px',
              border: 'none',
              background: 'linear-gradient(to right, var(--accent-hover), var(--accent-color))',
              color: 'white',
              fontSize: '1.1rem',
              fontWeight: 600,
              cursor: loading || !review.trim() ? 'not-allowed' : 'pointer',
              opacity: loading || !review.trim() ? 0.7 : 1,
              transition: 'transform 0.2s',
              boxShadow: '0 4px 15px rgba(139, 92, 246, 0.3)',
            }}
            onMouseOver={(e) => !loading && review.trim() && (e.target.style.transform = 'translateY(-2px)')}
            onMouseOut={(e) => e.target.style.transform = 'translateY(0)'}
          >
            {loading ? 'Analyzing...' : 'Analyze Sentiment'}
          </button>

          {error && (
            <div style={{ padding: '1rem', background: 'rgba(239, 68, 68, 0.1)', borderLeft: '4px solid var(--danger)', borderRadius: '8px', color: '#fca5a5' }}>
              Error: {error}
            </div>
          )}

          {prediction && (
            <div style={{
              marginTop: '1rem',
              padding: '2rem',
              borderRadius: '16px',
              background: 'rgba(0, 0, 0, 0.3)',
              border: '1px solid rgba(255, 255, 255, 0.05)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              animation: 'fadeIn 0.5s ease-out'
            }}>
              <div>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '0.5rem' }}>
                  Result
                </p>
                <h2 style={{ 
                  fontSize: '2.5rem', 
                  color: prediction === 'POSITIVE' ? 'var(--success)' : 'var(--danger)',
                  textShadow: `0 0 20px ${prediction === 'POSITIVE' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`
                }}>
                  {prediction}
                </h2>
              </div>
              
              {confidence && (
                <div style={{ textAlign: 'right' }}>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '0.5rem' }}>
                    Confidence
                  </p>
                  <h3 style={{ fontSize: '2rem', fontWeight: 300 }}>
                    {(confidence * 100).toFixed(1)}%
                  </h3>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </>
  );
}

export default App;
