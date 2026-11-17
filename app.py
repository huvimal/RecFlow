"""Local-only read-only recommendation API and browser demo."""
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI,HTTPException,Query
from fastapi.responses import HTMLResponse
from recflow import load,recommend,ROOT,PRIVATE,digest

@asynccontextmanager
async def lifespan(app):
    evidence=json.loads((ROOT/'reports/evaluation.json').read_text())
    for name in ('model.json','model.npz'):
        if digest(PRIVATE/name)!=evidence['artifacts'][name]:
            raise RuntimeError('Model does not match its evaluation evidence; retrain before serving')
    app.state.model=load()
    app.state.evidence=evidence
    yield

app=FastAPI(title='RecFlow',version='1.0.0',lifespan=lifespan)

@app.get('/health')
def health():return {'status':'ready','items':len(app.state.model['items'])}

@app.get('/users')
def users():return {'users':app.state.model['users'][:100],'limit':100}

@app.get('/recommend')
def recommendations(user:str=Query('new-buyer',max_length=80),k:int=Query(10,ge=1,le=50),
                    method:str=Query('collaborative',pattern='^(collaborative|content|popularity)$'),
                    query:str=Query('',max_length=160)):
    try:
        strategy,rows=recommend(app.state.model,user,k,method,query)
        return {'strategy':strategy,'recommendations':rows,'score_is_probability':False}
    except ValueError as e:raise HTTPException(422,str(e))

@app.get('/report')
def report():return app.state.evidence

@app.get('/',response_class=HTMLResponse)
def index():return PAGE

PAGE='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>RecFlow</title>
<style>body{margin:0;background:#091422;color:#e8f2fa;font:16px system-ui}main{max-width:1100px;margin:50px auto;padding:24px}h1{font-size:48px;margin:0}small{color:#98b0c3}.tag{color:#52e0b0}section{background:#122337;border:1px solid #29445e;border-radius:16px;padding:24px;margin-top:24px}label{display:inline-block;margin:10px 18px 10px 0}input,select,button{font:inherit;padding:10px;border-radius:8px;border:1px solid #39536e;background:#091422;color:#fff}button{background:#17a984;cursor:pointer}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:12px;border-bottom:1px solid #29445e}#status{color:#70e4c6}pre{white-space:pre-wrap;overflow-wrap:anywhere}.muted{color:#98b0c3}</style>
<main><p class="tag">LOCAL RECOMMENDATION ENGINE</p><h1>RecFlow</h1><p>Purchase history → learned retrieval → inspectable recommendations</p><small>UCI Online Retail · frozen chronological evaluation · local demo</small>
<section><h2>Explore recommendations</h2><label>Demo buyer<br><select id="user"><option value="new-buyer">New buyer (popularity fallback)</option></select></label><label>Ranking<br><select id="method"><option value="collaborative">Trained collaborative SVD</option><option value="content">Product title TF-IDF</option><option value="popularity">Popularity baseline</option></select></label><label>Optional product preference<br><input id="query" placeholder="e.g. heart bag light" maxlength="160"></label><button id="go">Recommend</button><p id="status" role="status"></p><table><thead><tr><th>Product</th><th>Score</th><th>Training buyers</th></tr></thead><tbody id="rows"></tbody></table><p class="muted">Previously purchased products are excluded. Scores are ranking values, not probabilities. A new buyer falls back to popularity; keyword preferences use title similarity.</p></section>
<section><h2>Measured test results</h2><table><thead><tr><th>Method</th><th>Recall@10</th><th>NDCG@10</th><th>Coverage</th></tr></thead><tbody id="metrics"></tbody></table><p class="muted">Offline future-purchase ranking. Training catalog only. New catalog items remain missed targets. These results do not measure conversion or revenue.</p><details><summary>Protocol and slices</summary><pre id="protocol"></pre></details></section><footer><p>Data: Chen, D. (2015), Online Retail, UCI · CC BY 4.0 · DOI 10.24432/C5BW33.</p></footer></main>
<script>
function cell(row,value){let c=document.createElement('td');c.textContent=value;row.appendChild(c)}
async function get(url){let r=await fetch(url);let data=await r.json();if(!r.ok)throw Error(data.detail||'Request failed');return data}
async function run(){let b=document.getElementById('go');b.disabled=true;try{let p=new URLSearchParams({user:document.getElementById('user').value,method:document.getElementById('method').value,query:document.getElementById('query').value,k:10});let d=await get('/recommend?'+p);document.getElementById('status').textContent='Strategy: '+d.strategy;let rows=document.getElementById('rows');rows.replaceChildren();for(let x of d.recommendations){let r=document.createElement('tr');cell(r,x.title);cell(r,x.score.toFixed(3));cell(r,x.training_buyers);rows.appendChild(r)}}catch(e){document.getElementById('status').textContent=e.message}finally{b.disabled=false}}
document.getElementById('go').onclick=run;
async function init(){try{let u=await get('/users');for(let id of u.users){let o=document.createElement('option');o.value=id;o.textContent=id;document.getElementById('user').appendChild(o)}let r=await get('/report');for(let [name,m] of Object.entries(r.test.methods)){let tr=document.createElement('tr');cell(tr,name);cell(tr,m.all.recall_at_10.toFixed(4));cell(tr,m.all.ndcg_at_10.toFixed(4));cell(tr,m.catalog_coverage.toFixed(3));document.getElementById('metrics').appendChild(tr)}document.getElementById('protocol').textContent=JSON.stringify({protocol:r.protocol,rows:r.rows,test:r.test},null,2);await run()}catch(e){document.getElementById('status').textContent=e.message}}init();
</script></html>'''
