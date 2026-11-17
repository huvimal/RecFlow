import json,os,platform,time
from datetime import datetime,timezone
from recflow import ROOT,PRIVATE,split,fit,save,evaluate,digest,TRAIN_END,VAL_END

def main():
    events=json.loads((PRIVATE/'events.json').read_text())
    train,val,test=split(events)
    start=time.perf_counter();model=fit(train);seconds=time.perf_counter()-start
    save(model)
    print('Training finished. Evaluating three ranking methods...',flush=True)
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),seed=42,rank=32,
        hardware=dict(platform=platform.platform(),cpu=platform.processor(),logical_cpus=os.cpu_count()),
        training_seconds=seconds,source=json.loads((PRIVATE/'data-source.json').read_text()),
        train_before=TRAIN_END,validation_before=VAL_END,
        rows=dict(train=len(train),validation=len(val),test=len(test)),
        training_users=len(model['users']),training_items=len(model['items']),
        protocol='Global time split; frozen train-only catalog/history. Macro user recall/NDCG@10 on future distinct products not bought in train. Cold catalog targets remain in denominator; repeated training purchases excluded. No validation tuning.',
        validation=evaluate(model,val),test=evaluate(model,test),
        artifacts={name:digest(PRIVATE/name) for name in ('events.json','model.json','model.npz')},
        code={name:digest(ROOT/name) for name in ('prepare.py','recflow.py','train.py')})
    out=ROOT/'reports';out.mkdir(exist_ok=True)
    (out/'evaluation.json').write_text(json.dumps(report,indent=2))
    lines=['# Measured results','','UCI Online Retail; CC BY 4.0. One frozen train-only fit. Offline purchase ranking, not revenue lift.','',
           '| Method | Test users | Recall@10 | NDCG@10 | Catalog coverage | p95 ms |','|---|---:|---:|---:|---:|---:|']
    for name,result in report['test']['methods'].items():
        m=result['all'];lines.append(f"| {name} | {m['users']} | {m['recall_at_10']:.4f} | {m['ndcg_at_10']:.4f} | {result['catalog_coverage']:.4f} | {result['latency_ms_p95']:.2f} |")
    lines+=['','See evaluation.json for warm/cold user slices, cold-item targets, popularity concentration, hardware and fingerprints.',
            'Catalog reflects products purchased in training, not real inventory availability. Unknown products cannot be recommended. Unknown users use popularity; explicit product keywords use title TF-IDF. No live deployment, business lift or privacy guarantee is claimed.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
