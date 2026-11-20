# Model comparison report

Test predictions evaluated: **3581**

| Metric | N-gram | Neural LM |
|---|---:|---:|
| Mean Reciprocal Rank | 0.1886 | 0.2093 |
| Hits@1 | 0.0958 | 0.0980 |
| Hits@3 | 0.1843 | 0.2092 |
| Hits@5 | 0.2435 | 0.2879 |
| Mean Rank | 19.6378 | 16.2080 |

Higher MRR and Hits@K are better; lower Mean Rank is better.
The dataset is synthetic and contains planted co-purchase itemsets.
