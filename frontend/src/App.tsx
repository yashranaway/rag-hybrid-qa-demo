import { useState } from 'react'

type QAResponse = {
  question: string
  before: string
  after: string
  sources: { text: string; score: number }[]
}

const apiUrl = 'http://127.0.0.1:8000/qa'

export default function App() {
  const samples = [
    'What is in front of the Notre Dame Main Building?',
    'Who first visited New York Harbor in 1524?',
    'What city became the last capital under the Articles of Confederation?'
  ]
  const [prompt, setPrompt] = useState(samples[0])
  const [result, setResult] = useState<QAResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [raw, setRaw] = useState<string>('')

  const ask = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(apiUrl, {
        method: 'POST',
        mode: 'cors',
        credentials: 'omit',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: prompt, top_k: 3 })
      })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || `Request failed with ${res.status}`)
      }
      const text = await res.text()
      setRaw(text)
      const data = JSON.parse(text) as QAResponse
      setResult(data)
      console.log('QA response', data)
    } finally {
      setLoading(false)
    }
  }

  const beforeText = (result?.before ?? '').trim().length ? result!.before : 'No answer returned.'
  const afterText = (result?.after ?? '').trim().length ? result!.after : 'No answer returned.'

  return (
    <div style={{
      minHeight: '100vh',
      background: '#0b0b0b',
      color: '#eaeaea',
      fontFamily: 'ui-sans-serif, system-ui, -apple-system',
      padding: '24px'
    }}>
      <style>{`
        @keyframes spin { from { transform: rotate(0deg);} to { transform: rotate(360deg);} }
        @keyframes pulse { 0% { opacity:.4;} 50% {opacity:1;} 100% {opacity:.4;} }
      `}</style>
      <h1 style={{textAlign:'center', fontWeight:600}}>Retrieval-Augmented Transformer Demo</h1>

      <div style={{display:'flex', gap:12, margin:'16px auto', width:'min(96vw, 1600px)'}}>
        <input
          value={prompt}
          onChange={(e)=>setPrompt(e.target.value)}
          onKeyDown={(e)=>{ if(e.key==='Enter' && !loading) ask() }}
          placeholder="Enter a question..."
          style={{flex:1, padding:'12px 14px', border:'1px solid #3a3a3a', background:'#111', color:'#fff', borderRadius:8}}
        />
        <button onClick={ask} disabled={loading}
          style={{padding:'12px 16px', background:'#fff', color:'#000', borderRadius:8, border:'none', cursor:'pointer'}}>
          {loading ? 'Thinking…' : 'Run'}
        </button>
      </div>

      <div style={{display:'flex', gap:8, flexWrap:'wrap', width:'min(96vw, 1600px)', margin:'0 auto 8px'}}>
        {samples.map((q)=> (
          <button key={q} onClick={()=>{ setPrompt(q); ask(); }}
            style={{padding:'6px 10px', background:'#1a1a1a', color:'#ddd', border:'1px solid #333', borderRadius:6, cursor:'pointer'}}>
            {q}
        </button>
        ))}
      </div>

      <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', width:'min(96vw, 1600px)', margin:'24px auto', gap:16}}>
        <div style={{border:'1px solid #3a3a3a', borderRadius:16, padding:20, minHeight:420, position:'relative'}}>
          <h2 style={{marginTop:0}}>Before (no RAG)</h2>
          {loading && (
            <div style={{position:'absolute', inset:0, background:'rgba(0,0,0,.35)', display:'flex', alignItems:'center', justifyContent:'center', borderRadius:16}}>
              <div style={{width:28, height:28, border:'3px solid #555', borderTopColor:'#fff', borderRadius:'50%', animation:'spin 1s linear infinite'}} />
            </div>
          )}
          <pre style={{whiteSpace:'pre-wrap'}}>{result ? beforeText : 'Ask a question to see the baseline answer.'}</pre>
        </div>
        <div style={{border:'1px solid #3a3a3a', borderRadius:16, padding:20, minHeight:420, position:'relative'}}>
          <h2 style={{marginTop:0}}>After (with RAG)</h2>
          {loading && (
            <div style={{position:'absolute', inset:0, background:'rgba(0,0,0,.35)', display:'flex', alignItems:'center', justifyContent:'center', borderRadius:16}}>
              <div style={{width:28, height:28, border:'3px solid #555', borderTopColor:'#fff', borderRadius:'50%', animation:'spin 1s linear infinite'}} />
            </div>
          )}
          <pre style={{whiteSpace:'pre-wrap'}}>{result ? afterText : 'Ask a question to see the RAG answer.'}</pre>
          <div style={{marginTop:16}}>
            <h3>Sources</h3>
            <ul style={{paddingLeft:18}}>
              {result?.sources?.map((s, i)=> (
                <li key={i} style={{marginBottom:8}}>
                  <small style={{opacity:.8}}>score {s.score.toFixed(3)}</small>
                  <div>{s.text.slice(0,200)}{s.text.length>200?'…':''}</div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      <div style={{width:'min(96vw, 1600px)', margin:'12px auto 0'}}>
        {error && (
          <div style={{color:'#ffb4b4', marginBottom:8}}>Error: {error}</div>
        )}
        {raw && (
          <details style={{opacity:.8}}>
            <summary>Show raw response</summary>
            <pre style={{whiteSpace:'pre-wrap'}}>{raw}</pre>
          </details>
        )}
      </div>
    </div>
  )
}
