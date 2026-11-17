"""Temporal recommendation research demo; no demographic features."""
import hashlib, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
PRIVATE = ROOT / 'private'
TRAIN_END = '2011-09-01'
VAL_END = '2011-10-15'

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def split(events):
    return ([e for e in events if e['date'] < TRAIN_END],
            [e for e in events if TRAIN_END <= e['date'] < VAL_END],
            [e for e in events if e['date'] >= VAL_END])

def tokenize(text):
    import re
    return set(re.findall(r'[a-z]{3,}', text.lower()))

def fit(events, seed=42, rank=32):
    users = sorted({e['user'] for e in events})
    items = sorted({e['item'] for e in events})
    if len(users) < 2 or len(items) < 2:
        raise ValueError('At least two training users and products required')
    ui, ii = {u:i for i,u in enumerate(users)}, {v:i for i,v in enumerate(items)}
    x = np.zeros((len(users),len(items)), dtype=np.float32)
    titles = {}
    for e in sorted(events,key=lambda v:v['date']):
        x[ui[e['user']],ii[e['item']]] = 1
        titles[e['item']] = e['title']
    popularity = x.sum(axis=0)
    weighted = x / np.sqrt(np.maximum(x.sum(axis=1,keepdims=True),1))
    # Randomized low-rank SVD of train-only binary purchase matrix.
    width = min(rank+8, min(x.shape))
    rng = np.random.default_rng(seed)
    q,_ = np.linalg.qr(weighted @ rng.normal(size=(len(items),width)).astype(np.float32))
    for _ in range(2):
        q,_ = np.linalg.qr(weighted @ (weighted.T @ q))
    _,s,vt = np.linalg.svd(q.T @ weighted,full_matrices=False)
    factors = vt[:rank].T * np.sqrt(s[:rank])
    factors /= np.maximum(np.linalg.norm(factors,axis=1,keepdims=True),1e-8)
    vocab = sorted(set().union(*(tokenize(titles[i]) for i in items)))
    vi = {v:i for i,v in enumerate(vocab)}
    content = np.zeros((len(items),len(vocab)),dtype=np.float32)
    for row,item in enumerate(items):
        for word in tokenize(titles[item]): content[row,vi[word]]=1
    idf = np.log((1+len(items))/(1+content.sum(axis=0)))+1
    content *= idf
    content /= np.maximum(np.linalg.norm(content,axis=1,keepdims=True),1e-8)
    return dict(users=users,items=items,titles=[titles[i] for i in items],vocab=vocab,
                seen=x,factors=factors.astype(np.float32),content=content,idf=idf,popularity=popularity)

def recommend(model,user,k=10,method='collaborative',query=''):
    if method not in ('collaborative','content','popularity'):raise ValueError('Unknown model')
    if not 1 <= k <= 50:raise ValueError('k must be 1..50')
    users=model['users']; row=users.index(user) if user in users else None
    seen=np.zeros(len(model['items']),dtype=bool) if row is None else model['seen'][row]>0
    if query.strip():
        vec=np.array([float(w in tokenize(query)) for w in model['vocab']])*model['idf']
        if not np.any(vec):raise ValueError('No known product words; try BAG, HEART or LIGHT')
        scores=model['content']@vec; strategy='content query (explicit preference)'
    elif method=='popularity' or row is None:
        scores=model['popularity'].copy();strategy='popularity fallback' if row is None else 'popularity'
    else:
        vectors=model['factors'] if method=='collaborative' else model['content']
        profile=vectors[seen].mean(axis=0)
        scores=vectors@profile;strategy=method
    scores=np.array(scores,dtype=float);scores[seen]=-np.inf
    order=np.lexsort((np.arange(len(scores)),-scores))
    order=[int(i) for i in order if np.isfinite(scores[i])][:k]
    return strategy,[{'product_id':model['items'][i],'title':model['titles'][i],
                     'score':float(scores[i]),'training_buyers':int(model['popularity'][i])} for i in order]

def metrics(ranked,targets,k=10):
    selected=ranked[:k]; hits=[int(i in targets) for i in selected]
    recall=sum(hits)/len(targets) if targets else 0.0
    dcg=sum(h/np.log2(pos+2) for pos,h in enumerate(hits))
    ideal=sum(1/np.log2(pos+2) for pos in range(min(k,len(targets))))
    return recall,dcg/ideal if ideal else 0.0

def evaluate(model,events,k=10):
    targets={}
    for e in events:targets.setdefault(e['user'],set()).add(e['item'])
    catalog=set(model['items']);ui={u:i for i,u in enumerate(model['users'])}
    for u in list(targets):
        if u in ui:
            targets[u]-={model['items'][i] for i in np.flatnonzero(model['seen'][ui[u]])}
        if not targets[u]:del targets[u]
    results={}
    for method in ('popularity','content','collaborative'):
        slices={s:[] for s in ('all','warm_user','cold_user')};covered=set();latencies=[];head=[]
        threshold=np.quantile(model['popularity'],.8)
        for u,truth in sorted(targets.items()):
            start=time.perf_counter();_,rows=recommend(model,u,k,method)
            latencies.append((time.perf_counter()-start)*1000)
            ids=[r['product_id'] for r in rows];covered.update(ids)
            head.extend(r['training_buyers']>=threshold for r in rows)
            value=metrics(ids,truth,k);slices['all'].append(value)
            slices['warm_user' if u in ui else 'cold_user'].append(value)
        results[method]={s:{'users':len(vals),'recall_at_10':float(np.mean([v[0] for v in vals])) if vals else None,
                            'ndcg_at_10':float(np.mean([v[1] for v in vals])) if vals else None} for s,vals in slices.items()}
        results[method].update(catalog_coverage=len(covered)/len(catalog),
            head_item_share=float(np.mean(head)) if head else None,
            latency_ms_p50=float(np.percentile(latencies,50)) if latencies else None,
            latency_ms_p95=float(np.percentile(latencies,95)) if latencies else None)
    return {'methods':results,'target_users':len(targets),
            'unseen_target_pairs':sum(map(len,targets.values())),
            'cold_item_target_pairs':sum(len(v-catalog) for v in targets.values())}

def save(model):
    PRIVATE.mkdir(exist_ok=True)
    meta={k:model[k] for k in ('users','items','titles','vocab')}
    (PRIVATE/'model.json').write_text(json.dumps(meta),encoding='utf-8')
    np.savez_compressed(PRIVATE/'model.npz',**{k:model[k] for k in ('seen','factors','content','idf','popularity')})

def load():
    model=json.loads((PRIVATE/'model.json').read_text(encoding='utf-8'))
    with np.load(PRIVATE/'model.npz',allow_pickle=False) as arrays:
        model.update({k:arrays[k] for k in arrays.files})
    return model
