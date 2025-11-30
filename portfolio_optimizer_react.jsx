import React, { useState, useMemo } from "react";
import Papa from "papaparse";
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Line } from "recharts";
import { motion } from "framer-motion";

// PortfolioOptimizer.jsx
// Single-file React component that provides a clean Tailwind UI, CSV upload, Monte Carlo
// portfolio simulation, and visualization (efficient frontier + scatter of portfolios).
// - Uses PapaParse for CSV parsing and Recharts for charts.
// - Monte Carlo approach avoids QP solver and runs entirely in-browser.

export default function PortfolioOptimizer() {
  const [tickers, setTickers] = useState([]); // array of strings
  const [returnsData, setReturnsData] = useState([]); // array of arrays: rows = dates, cols = tickers
  const [numSims, setNumSims] = useState(5000);
  const [riskFree, setRiskFree] = useState(0.02);
  const [allowShort, setAllowShort] = useState(false);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  // Basic CSV expected format: first column is date (ignored), subsequent columns are prices or returns per ticker.
  // If prices are provided we convert to log returns. If returns already provided, the user can upload returns.

  function parseCSV(file) {
    setLoading(true);
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (res) => {
        // res.data is an array of objects keyed by column headers
        if (!res || !res.data || res.data.length === 0) {
          setLoading(false);
          return;
        }
        const headers = Object.keys(res.data[0]);
        // If first header is "Date" or similar, drop it
        const potentialDate = headers[0].toLowerCase();
        const dataHeaders = headers.filter(h => h && !/^date$/i.test(h));
        // Build matrix of price series
        const cols = dataHeaders.map(col => res.data.map(r => (r[col] == null ? NaN : r[col])));

        // Check if values look like prices (>1 maybe) or returns (-1..1)
        const flattened = cols.flat().filter(v => Number.isFinite(v));
        const avg = flattened.reduce((a,b)=>a+b,0)/flattened.length;
        const isPrice = Math.abs(avg) > 1.5 || flattened.some(v => Math.abs(v) > 1.5);

        // Convert prices -> daily returns (log returns)
        const returns = cols.map(series => {
          const out = [];
          for (let i = 1; i < series.length; i++) {
            const a = series[i-1];
            const b = series[i];
            if (!Number.isFinite(a) || !Number.isFinite(b) || a <= 0 || b <= 0) {
              out.push(NaN);
            } else {
              out.push(Math.log(b / a));
            }
          }
          return out;
        });

        // Align rows and drop NaN rows across any column
        const rowCount = Math.max(...returns.map(r => r.length));
        const cleaned = [];
        for (let r = 0; r < rowCount; r++) {
          const row = returns.map(col => col[r]);
          if (row.every(v => Number.isFinite(v))) cleaned.push(row);
        }

        // transpose cleaned to columns
        const finalCols = dataHeaders.map((_, c) => cleaned.map(row => row[c]));

        setTickers(dataHeaders);
        setReturnsData(finalCols);
        setLoading(false);
      },
      error: (err) => {
        console.error(err);
        setLoading(false);
      }
    });
  }

  // Helper numeric functions
  function mean(arr) {
    return arr.reduce((a,b) => a+b,0)/arr.length;
  }
  function std(arr) {
    const m = mean(arr);
    const v = arr.reduce((s,x)=>s+(x-m)*(x-m),0)/(arr.length-1);
    return Math.sqrt(v);
  }
  function covMatrix(cols) {
    const n = cols.length;
    const m = cols[0].length;
    const means = cols.map(c => mean(c));
    const cov = Array.from({length:n}, ()=>Array(n).fill(0));
    for (let i=0;i<n;i++){
      for (let j=i;j<n;j++){
        let s = 0;
        for (let k=0;k<m;k++) s += (cols[i][k]-means[i])*(cols[j][k]-means[j]);
        const val = s/(m-1);
        cov[i][j]=val; cov[j][i]=val;
      }
    }
    return cov;
  }

  function dot(a,b){
    return a.reduce((s,x,i)=>s + x*b[i],0);
  }

  function portfolioStats(weights, expReturns, cov) {
    const er = dot(weights, expReturns);
    // variance = w^T * cov * w
    const n = weights.length;
    let varp = 0;
    for (let i=0;i<n;i++) for (let j=0;j<n;j++) varp += weights[i]*weights[j]*cov[i][j];
    return { return: er, volatility: Math.sqrt(varp) };
  }

  // Monte Carlo simulation to find many portfolios and approximate efficient frontier
  function runMonteCarlo({ sims = 3000, allowShorting=false, rf = 0.02 }){
    if (returnsData.length === 0) return;
    setLoading(true);
    // Calculate expected returns (annualize assuming 252 trading days)
    const daysPerYear = 252;
    const expReturns = returnsData.map(col => mean(col) * daysPerYear);
    const cov = covMatrix(returnsData).map(row => row.map(v => v * daysPerYear));

    const n = returnsData.length;
    const records = [];
    for (let s=0;s<sims;s++){
      // generate random weights
      let w = Array.from({length:n}, ()=>Math.random());
      if (allowShort) {
        // allow negative weights by shifting
        w = w.map(v => (v - 0.5) * 2); // -1..1
      }
      // enforce sum to 1
      const sum = w.reduce((a,b)=>a+b,0) || 1;
      w = w.map(x => x/sum);

      const st = portfolioStats(w, expReturns, cov);
      const sharpe = (st.return - rf)/st.volatility;
      records.push({ weights: w, ret: st.return, vol: st.volatility, sharpe });
    }

    // find max sharpe and min vol
    const maxSharpe = records.reduce((a,b)=> b.sharpe > a.sharpe ? b : a, records[0]);
    const minVol = records.reduce((a,b)=> b.vol < a.vol ? b : a, records[0]);

    // approximate efficient frontier by selecting portfolios with lowest vol for bins of return
    records.sort((a,b) => a.ret - b.ret);
    const bins = 60;
    const frontier = [];
    for (let i=0;i<bins;i++){
      const lo = Math.floor((i/ bins) * records.length);
      const hi = Math.floor(((i+1)/bins) * records.length) - 1;
      const slice = records.slice(lo, Math.max(lo,hi)+1);
      if (slice.length===0) continue;
      const best = slice.reduce((p,c)=> c.vol < p.vol ? c : p, slice[0]);
      frontier.push({ ret: best.ret, vol: best.vol });
    }

    setResults({ records, maxSharpe, minVol, frontier, expReturns, cov });
    setLoading(false);
  }

  // small utility to display allocation for a weights array
  function allocationTable(weights) {
    return tickers.map((t,i)=>({ ticker: t, weight: +(weights[i]*100).toFixed(2) }));
  }

  function downloadCSV() {
    if (!results) return;
    const header = ["ret","vol","sharpe", ...tickers].join(",");
    const lines = results.records.map(r=>{
      const w = r.weights.map(x=>x.toFixed(6)).join(",");
      return [r.ret.toFixed(6), r.vol.toFixed(6), r.sharpe.toFixed(6), w].join(",");
    });
    const csv = [header, ...lines].join("\n");
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'portfolio_simulation.csv';
    a.click();
    URL.revokeObjectURL(url);
  }

  // Example: quick demo data (3 assets)
  function loadDemo() {
    // create synthetic returns for 3 assets
    const days = 252*3; // 3 years
    const mu = [0.08/252, 0.12/252, 0.05/252];
    const sigma = [0.12/Math.sqrt(252), 0.20/Math.sqrt(252), 0.08/Math.sqrt(252)];
    const cols = mu.map((m,i)=> Array.from({length:days}, ()=> m + randn() * sigma[i]));
    setTickers(['Asset A','Asset B','Asset C']);
    setReturnsData(cols);
  }
  function randn(){
    // Box-Muller
    let u=0,v=0; while(u===0) u=Math.random(); while(v===0) v=Math.random();
    return Math.sqrt(-2.0*Math.log(u))*Math.cos(2*Math.PI*v);
  }

  const chartData = useMemo(()=>{
    if (!results) return [];
    return results.records.map(r=>({ x: r.vol, y: r.ret, sharpe: r.sharpe }));
  }, [results]);

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-6xl mx-auto">
        <motion.div initial={{opacity:0, y:10}} animate={{opacity:1, y:0}} className="bg-white shadow rounded-2xl p-6">
          <h1 className="text-2xl font-semibold">Portfolio Optimizer</h1>
          <p className="text-sm text-gray-500 mt-1">Upload price/return CSV or load demo data. Run Monte Carlo to approximate the efficient frontier.</p>

          <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="col-span-1">
              <label className="block text-xs font-medium text-gray-700">Upload CSV</label>
              <input type="file" accept=".csv" className="mt-2" onChange={(e)=>{ if (e.target.files[0]) parseCSV(e.target.files[0]); }} />

              <div className="mt-3">
                <button className="px-4 py-2 rounded-xl bg-blue-600 text-white mr-2" onClick={()=>loadDemo()}>Load Demo Data</button>
                <button className="px-4 py-2 rounded-xl bg-green-600 text-white" onClick={()=>runMonteCarlo({sims: numSims, allowShorting: allowShort, rf: riskFree})}>Run Simulation</button>
              </div>

              <div className="mt-4">
                <label className="text-xs">Number of simulations: <strong>{numSims}</strong></label>
                <input type="range" min={500} max={20000} step={100} value={numSims} onChange={(e)=>setNumSims(+e.target.value)} className="w-full" />
                <label className="text-xs">Risk-free rate (annual): <strong>{(riskFree*100).toFixed(2)}%</strong></label>
                <input type="range" min={0} max={0.1} step={0.001} value={riskFree} onChange={(e)=>setRiskFree(+e.target.value)} className="w-full" />

                <div className="flex items-center mt-2">
                  <input type="checkbox" checked={allowShort} onChange={(e)=>setAllowShort(e.target.checked)} id="short" />
                  <label htmlFor="short" className="ml-2 text-sm">Allow shorting</label>
                </div>

                {loading && <div className="mt-3 text-sm text-gray-600">Working... this runs in your browser.</div>}
              </div>
            </div>

            <div className="col-span-2 md:col-span-2">
              <div className="h-96 bg-gray-100 rounded-xl p-3">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid />
                    <XAxis dataKey="x" name="Volatility" unit="" type="number" domain={[0, 'dataMax']} />
                    <YAxis dataKey="y" name="Return" unit="" type="number" domain={["dataMin","dataMax"]} />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    {results && (
                      <>
                        <Scatter name="Portfolios" data={chartData} fill="#4f46e5" opacity={0.6} />
                        {/* efficient frontier line */}
                        <Line type="monotone" dataKey="y" data={results.frontier.map(p=>({x: p.vol, y: p.ret}))} stroke="#10b981" dot={false} strokeWidth={2} />
                      </>
                    )}
                  </ScatterChart>
                </ResponsiveContainer>
              </div>

              <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-3">
                <div className="bg-white rounded-lg p-3 shadow">
                  <div className="text-xs text-gray-500">Tickers</div>
                  <div className="font-medium mt-1">{tickers.length ? tickers.join(', ') : 'No data loaded'}</div>
                </div>
                <div className="bg-white rounded-lg p-3 shadow">
                  <div className="text-xs text-gray-500">Best Sharpe</div>
                  <div className="font-medium mt-1">{results ? `${(results.maxSharpe.sharpe).toFixed(3)} (ret ${(results.maxSharpe.ret*100).toFixed(2)}%, vol ${(results.maxSharpe.vol*100).toFixed(2)}%)` : '—'}</div>
                </div>
                <div className="bg-white rounded-lg p-3 shadow">
                  <div className="text-xs text-gray-500">Min Vol</div>
                  <div className="font-medium mt-1">{results ? `${(results.minVol.vol*100).toFixed(2)}% vol` : '—'}</div>
                </div>
              </div>

            </div>
          </div>

          {/* Results and allocation */}
          <div className="mt-6">
            {results && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white rounded-xl p-4 shadow">
                  <h3 className="font-semibold">Max Sharpe Portfolio Allocation</h3>
                  <table className="w-full mt-3 text-sm table-auto">
                    <thead><tr><th className="text-left">Ticker</th><th className="text-right">Weight %</th></tr></thead>
                    <tbody>
                      {allocationTable(results.maxSharpe.weights).map(row=> (
                        <tr key={row.ticker}><td className="py-1">{row.ticker}</td><td className="py-1 text-right">{row.weight}%</td></tr>
                      ))}
                    </tbody>
                  </table>

                  <div className="mt-3 flex gap-2">
                    <button className="px-3 py-1 rounded-md bg-indigo-600 text-white" onClick={()=>downloadCSV()}>Download simulation CSV</button>
                  </div>
                </div>

                <div className="bg-white rounded-xl p-4 shadow">
                  <h3 className="font-semibold">Min Vol Portfolio Allocation</h3>
                  <table className="w-full mt-3 text-sm table-auto">
                    <thead><tr><th className="text-left">Ticker</th><th className="text-right">Weight %</th></tr></thead>
                    <tbody>
                      {allocationTable(results.minVol.weights).map(row=> (
                        <tr key={row.ticker}><td className="py-1">{row.ticker}</td><td className="py-1 text-right">{row.weight}%</td></tr>
                      ))}
                    </tbody>
                  </table>

                  <div className="mt-3 text-xs text-gray-500">Tip: Use the sliders and allow shorting if you want to explore constrained/unconstrained results.</div>
                </div>
              </div>
            )}
          </div>

          <div className="mt-6 text-xs text-gray-400">Built with React + Tailwind + Recharts + PapaParse. This runs fully in your browser — no server required. For a full production app, add input validation, authentication, and optionally a backend for large datasets.</div>

        </motion.div>
      </div>
    </div>
  );
}
