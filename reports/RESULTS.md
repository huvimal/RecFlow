# Measured results

UCI Online Retail; CC BY 4.0. One frozen train-only fit. Offline purchase ranking, not revenue lift.

| Method | Test users | Recall@10 | NDCG@10 | Catalog coverage | p95 ms |
|---|---:|---:|---:|---:|---:|
| popularity | 2213 | 0.0180 | 0.0462 | 0.0239 | 7.18 |
| content | 2213 | 0.0340 | 0.0696 | 0.5827 | 7.63 |
| collaborative | 2213 | 0.0227 | 0.0468 | 0.7281 | 7.15 |

See evaluation.json for warm/cold user slices, cold-item targets, popularity concentration, hardware and fingerprints.
Catalog reflects products purchased in training, not real inventory availability. Unknown products cannot be recommended. Unknown users use popularity; explicit product keywords use title TF-IDF. No live deployment, business lift or privacy guarantee is claimed.
