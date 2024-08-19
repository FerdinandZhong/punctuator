## Llama3.1-8b-Instruct

#### Restoration with roberta output
```
2024-08-19 01:48:55,849 - INFO - pipeline.py:253 - pipeline.evaluate_llm_output - 97651 - total not matched index: 23
2024-08-19 01:48:56,834 - INFO - pipeline.py:274 - pipeline.evaluate_llm_output - 97651 - validation report for meta-llama/Meta-Llama-3.1-8B-Instruct with bert result matched: 
                  precision    recall  f1-score   support

          COMMA     0.4138    0.6000    0.4898     42627
         PERIOD     0.5800    0.4334    0.4961     33882
   QUESTIONMARK     0.5973    0.4554    0.5168      3061
EXCLAMATIONMARK     0.4000    0.1115    0.1743       323
          PUNCT     1.0000    1.0000    1.0000         0

      micro avg     0.4655    0.5218    0.4921     79893
      macro avg     0.5982    0.5200    0.5354     79893
   weighted avg     0.4913    0.5218    0.4922     79893

2024-08-19 01:48:56,835 - INFO - pipeline.py:279 - pipeline.evaluate_llm_output - 97651 - not matched avg similarity score: 0.591
```

#### Direct restoration 

```
2024-08-19 10:08:54,404 - INFO - pipeline.py:253 - pipeline.evaluate_llm_output - 97651 - total not matched index: 6
2024-08-19 10:08:55,642 - INFO - pipeline.py:274 - pipeline.evaluate_llm_output - 97651 - validation report for meta-llama/Meta-Llama-3.1-8B-Instruct with bert result matched: 
                  precision    recall  f1-score   support

          COMMA     0.4103    0.4206    0.4154     54113
         PERIOD     0.4518    0.4438    0.4478     42738
   QUESTIONMARK     0.4655    0.4146    0.4386      3927
EXCLAMATIONMARK     0.1714    0.0402    0.0651       448
          PUNCT     1.0000    1.0000    1.0000         0

      micro avg     0.4292    0.4285    0.4288    101226
      macro avg     0.4998    0.4638    0.4734    101226
   weighted avg     0.4289    0.4285    0.4284    101226

2024-08-19 10:08:55,643 - INFO - pipeline.py:279 - pipeline.evaluate_llm_output - 97651 - not matched avg similarity score: 0.885
```

## Llama3.1-70b-Instruct
