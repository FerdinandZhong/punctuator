## Llama3.1-8b-Instruct

### Restoration with roberta output
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

### Direct restoration 

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

### Restoration with roberta output

```
2024-08-19 18:13:13,112 - INFO - pipeline.py:253 - pipeline.evaluate_llm_output - 97651 - total not matched index: 18
2024-08-19 18:13:13,813 - INFO - pipeline.py:274 - pipeline.evaluate_llm_output - 97651 - validation report for meta-llama/Meta-Llama-3.1-70B-Instruct with bert result matched: 
                  precision    recall  f1-score   support

          COMMA     0.4271    0.5933    0.4966     53560
         PERIOD     0.5563    0.5047    0.5292     42320
   QUESTIONMARK     0.5524    0.5297    0.5408      3900
EXCLAMATIONMARK     0.1994    0.1492    0.1707       449
          PUNCT     1.0000    1.0000    1.0000         0

      micro avg     0.4729    0.5514    0.5091    100229
      macro avg     0.5470    0.5554    0.5475    100229
   weighted avg     0.4855    0.5514    0.5107    100229

2024-08-19 18:13:13,813 - INFO - pipeline.py:279 - pipeline.evaluate_llm_output - 97651 - not matched avg similarity score: 0.480
```

### Direct restoration 

```
2024-08-19 18:37:54,524 - INFO - pipeline.py:253 - pipeline.evaluate_llm_output - 97651 - total not matched index: 7
2024-08-19 18:37:55,187 - INFO - pipeline.py:274 - pipeline.evaluate_llm_output - 97651 - validation report for meta-llama/Meta-Llama-3.1-70B-Instruct with bert result matched: 
                  precision    recall  f1-score   support

          COMMA     0.4058    0.4773    0.4387     54127
         PERIOD     0.4877    0.4525    0.4694     42766
   QUESTIONMARK     0.4834    0.4686    0.4759      3944
EXCLAMATIONMARK     0.1470    0.1341    0.1402       455
          PUNCT     1.0000    1.0000    1.0000         0

      micro avg     0.4377    0.4650    0.4509    101292
      macro avg     0.5048    0.5065    0.5048    101292
   weighted avg     0.4422    0.4650    0.4518    101292

2024-08-19 18:37:55,188 - INFO - pipeline.py:279 - pipeline.evaluate_llm_output - 97651 - not matched avg similarity score: 0.998
```

## Qwen2-72B-Instruct

### Restoration with roberta output

```
2024-08-19 19:10:47,757 - INFO - pipeline.py:253 - pipeline.evaluate_llm_output - 97651 - total not matched index: 7
2024-08-19 19:10:48,990 - INFO - pipeline.py:274 - pipeline.evaluate_llm_output - 97651 - validation report for Qwen/Qwen2-72B-Instruct with bert result matched: 
                  precision    recall  f1-score   support

          COMMA     0.4958    0.6865    0.5757     54210
         PERIOD     0.6648    0.5882    0.6242     42840
   QUESTIONMARK     0.7128    0.5008    0.5883      3946
EXCLAMATIONMARK     0.1333    0.0088    0.0164       457
          PUNCT     1.0000    1.0000    1.0000         0

      micro avg     0.5562    0.6347    0.5929    101453
      macro avg     0.6014    0.5568    0.5609    101453
   weighted avg     0.5740    0.6347    0.5942    101453

2024-08-19 19:10:48,991 - INFO - pipeline.py:279 - pipeline.evaluate_llm_output - 97651 - not matched avg similarity score: 0.995
```

### Direct restoration 

```
2024-08-19 19:24:10,268 - INFO - pipeline.py:253 - pipeline.evaluate_llm_output - 97651 - total not matched index: 7
2024-08-19 19:24:11,441 - INFO - pipeline.py:274 - pipeline.evaluate_llm_output - 97651 - validation report for Qwen/Qwen2-72B-Instruct with bert result matched: 
                  precision    recall  f1-score   support

          COMMA     0.5346    0.5172    0.5258     54220
         PERIOD     0.6341    0.5774    0.6044     42830
   QUESTIONMARK     0.6620    0.5887    0.6232      3936
EXCLAMATIONMARK     0.3333    0.0722    0.1187       457
          PUNCT     1.0000    1.0000    1.0000         0

      micro avg     0.5799    0.5434    0.5611    101443
      macro avg     0.6328    0.5511    0.5744    101443
   weighted avg     0.5806    0.5434    0.5609    101443

2024-08-19 19:24:11,441 - INFO - pipeline.py:279 - pipeline.evaluate_llm_output - 97651 - not matched avg similarity score: 0.999
```

## Llama3.1-405b-Instruct

### Restoration with roberta output
```
2024-08-19 22:29:39,121 - INFO - pipeline.py:253 - pipeline.evaluate_llm_output - 97651 - total not matched index: 3
2024-08-19 22:29:40,334 - INFO - pipeline.py:274 - pipeline.evaluate_llm_output - 97651 - validation report for meta-llama/Meta-Llama-3.1-405B-Instruct with bert result matched: 
                  precision    recall  f1-score   support

          COMMA     0.3877    0.4916    0.4335     53746
         PERIOD     0.4828    0.4361    0.4582     42498
   QUESTIONMARK     0.4687    0.4559    0.4622      3893
EXCLAMATIONMARK     0.1810    0.1774    0.1792       451
          PUNCT     1.0000    1.0000    1.0000         0

      micro avg     0.4226    0.4654    0.4429    100588
      macro avg     0.5040    0.5122    0.5066    100588
   weighted avg     0.4301    0.4654    0.4439    100588

2024-08-19 22:29:40,335 - INFO - pipeline.py:279 - pipeline.evaluate_llm_output - 97651 - not matched avg similarity score: 0.997
```

### Direct restoration 

```
2024-08-19 22:50:01,099 - INFO - pipeline.py:253 - pipeline.evaluate_llm_output - 97651 - total not matched index: 4
2024-08-19 22:50:02,374 - INFO - pipeline.py:274 - pipeline.evaluate_llm_output - 97651 - validation report for meta-llama/Meta-Llama-3.1-405B-Instruct with bert result matched: 
                  precision    recall  f1-score   support

          COMMA     0.4529    0.5141    0.4816     54334
         PERIOD     0.5387    0.5090    0.5234     42931
   QUESTIONMARK     0.5457    0.5246    0.5349      3950
EXCLAMATIONMARK     0.2403    0.1225    0.1623       457
          PUNCT     1.0000    1.0000    1.0000         0

      micro avg     0.4885    0.5106    0.4993    101672
      macro avg     0.5555    0.5340    0.5404    101672
   weighted avg     0.4918    0.5106    0.4999    101672

2024-08-19 22:50:02,375 - INFO - pipeline.py:279 - pipeline.evaluate_llm_output - 97651 - not matched avg similarity score: 0.999
```