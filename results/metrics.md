# Test set results (2375 English tickets)

| Task | Classes | Baseline acc | Baseline macro-F1 | DistilBERT acc | DistilBERT macro-F1 |
|---|---:|---:|---:|---:|---:|
| queue | 10 | 0.613 | 0.541 | **0.674** | **0.673** |
| priority | 3 | 0.666 | 0.645 | **0.719** | **0.711** |
| type | 4 | 0.885 | 0.888 | **0.897** | **0.904** |

## queue

```
                                 precision    recall  f1-score   support

              Technical Support      0.664     0.730     0.695       686
                Product Support      0.644     0.621     0.632       443
               Customer Service      0.584     0.613     0.598       357
                     IT Support      0.624     0.620     0.622       284
           Billing and Payments      0.867     0.835     0.851       242
          Returns and Exchanges      0.673     0.632     0.652       117
Service Outages and Maintenance      0.782     0.656     0.713        93
            Sales and Pre-Sales      0.855     0.644     0.734        73
                Human Resources      0.727     0.696     0.711        46
                General Inquiry      0.812     0.382     0.520        34

                       accuracy                          0.674      2375
                      macro avg      0.723     0.643     0.673      2375
                   weighted avg      0.678     0.674     0.674      2375

```

## priority

```
              precision    recall  f1-score   support

         low      0.711     0.636     0.672       492
      medium      0.701     0.739     0.719       969
        high      0.744     0.743     0.743       914

    accuracy                          0.719      2375
   macro avg      0.719     0.706     0.711      2375
weighted avg      0.719     0.719     0.719      2375

```

## type

```
              precision    recall  f1-score   support

    Incident      0.874     0.881     0.877       946
     Request      0.996     0.988     0.992       683
     Problem      0.776     0.767     0.771       510
      Change      0.967     0.983     0.975       236

    accuracy                          0.897      2375
   macro avg      0.903     0.905     0.904      2375
weighted avg      0.897     0.897     0.897      2375

```
