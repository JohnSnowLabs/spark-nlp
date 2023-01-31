---
layout: model
title: Detect Entities in 40 languages - XTREME (ner_xtreme_glove_840B_300)
author: John Snow Labs
name: ner_xtreme_glove_840B_300
date: 2021-07-19
tags: [open_source, xtreme, ner, multilingual, xx, glove]
task: Named Entity Recognition
language: xx
edition: Spark NLP 3.1.3
spark_version: 2.4
supported: true
annotator: NerDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization. This NER model was trained over the XTREME dataset by using WordEmbeddings (glove_840B_300).

This NER model covers a subset of the 40 languages included in XTREME (shown here with their ISO 639-1 code): 

`af`, `ar`, `bg`, `bn`, `de`, `el`, `en`, `es`, `et`, `eu`, `fa`, `fi`, `fr`, `he`, `hi`, `hu`, `id`, `it`, `ja`, `jv`, `ka`, `kk`, `ko`, `ml`, `mr`, `ms`, `my`, `nl`, `pt`, `ru`, `sw`, `ta`, `te`, `th`, `tl`, `tr`, `ur`, `vi`, `yo`, and `zh`

## Predicted Entities

- B-LOC
- I-LOC
- B-ORG
- I-ORG
- B-PER
- I-PER

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_xtreme_glove_840B_300_xx_3.1.3_2.4_1626709939135.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ner_xtreme_glove_840B_300_xx_3.1.3_2.4_1626709939135.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

embeddings = WordEmbeddingsModel\
    .pretrained('glove_840B_300', 'xx')\
    .setInputCols(["token", "document"])\
    .setOutputCol("embeddings")

ner_model = NerDLModel.pretrained('ner_xtreme_glove_840B_300', 'xx') \
    .setInputCols(['document', 'token', 'embeddings']) \
    .setOutputCol('ner')

ner_converter = NerConverter() \
    .setInputCols(['document', 'token', 'ner']) \
    .setOutputCol('entities')

pipeline = Pipeline(stages=[
document_assembler, 
tokenizer,
embeddings,
ner_model,
ner_converter
])

text_list = [["""Jerome Horsey was a resident of the Russia Company in Moscow from 1572 to 1585."""],
            ["""Emilie Hartmanns Vater August Hartmann war Lehrer an der Hohen Karlsschule in Stuttgart, bis zu deren Auflösung 1793."""],
             ["""James Watt nacque in Scozia il 19 gennaio 1736 da genitori presbiteriani."""],
             ["""Quand j'ai dit à John que je voulais déménager en Alaska, il m'a prévenu que j'aurais du mal à trouver un Starbucks là-bas."""]]

example = spark.createDataFrame(text_list).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val tokenizer = new Tokenizer() 
    .setInputCols("document") 
    .setOutputCol("token")

val embeddings = WordEmbeddingsModel.pretrained("glove_840B_300", "xx")
    .setInputCols(Array("document", "token"))
    .setOutputCol("embeddings")

val ner_model = NerDLModel.pretrained("ner_xtreme_glove_840B_300", "xx") 
    .setInputCols(Array("document", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = NerConverter() 
    .setInputCols(Array("document", "token", "ner")) 
    .setOutputCol("entities")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, embeddings, ner_model, ner_converter))

val data = Seq(("""Jerome Horsey was a resident of the Russia Company in Moscow from 1572 to 1585."""),
            ("""Emilie Hartmanns Vater August Hartmann war Lehrer an der Hohen Karlsschule in Stuttgart, bis zu deren Auflösung 1793."""),
             ("""James Watt nacque in Scozia il 19 gennaio 1736 da genitori presbiteriani."""),
             ("""Quand j'ai dit à John que je voulais déménager en Alaska, il m'a prévenu que j'aurais du mal à trouver un Starbucks là-bas.""")).toDS.toDF("text"))

val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = [["""Jerome Horsey was a resident of the Russia Company in Moscow from 1572 to 1585."""],
            ["""Emilie Hartmanns Vater August Hartmann war Lehrer an der Hohen Karlsschule in Stuttgart, bis zu deren Auflösung 1793."""],
             ["""James Watt nacque in Scozia il 19 gennaio 1736 da genitori presbiteriani."""],
             ["""Quand j'ai dit à John que je voulais déménager en Alaska, il m'a prévenu que j'aurais du mal à trouver un Starbucks là-bas."""]]

ner_df = nlu.load('xx.ner.ner_xtreme_glove_840B_300').predict(text, output_level='token')
```
</div>

## Results
```bash
+--------------+---------+
|chunk         |ner_label|
+--------------+---------+
|Jerome Horsey |PER      |
|Russia Company|ORG      |
|Stuttgart     |LOC      |
|Scozia        |LOC      |
|John          |LOC      |
|Alaska        |LOC      |
|Starbucks     |ORG      |
+--------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_xtreme_glove_840B_300|
|Type:|ner|
|Compatibility:|Spark NLP 3.1.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|xx|

## Data Source

[https://github.com/google-research/xtreme](https://github.com/google-research/xtreme)

## Benchmarking

```bash
Language by language benchmarks (multi-label classification and CoNLL Eval):

###############################
lang:  af
precision    recall  f1-score   support

B-LOC       0.83      0.84      0.83       562
I-ORG       0.88      0.87      0.87       786
I-LOC       0.70      0.60      0.65       198
I-PER       0.90      0.91      0.91       504
B-ORG       0.87      0.81      0.84       569
B-PER       0.90      0.89      0.89       356

micro avg       0.86      0.85      0.85      2975
macro avg       0.84      0.82      0.83      2975
weighted avg       0.86      0.85      0.85      2975

processed 10808 tokens with 1487 phrases; found: 1460 phrases; correct: 1230.
accuracy:  84.50%; (non-O)
accuracy:  94.64%; precision:  84.25%; recall:  82.72%; FB1:  83.47
LOC: precision:  81.15%; recall:  82.74%; FB1:  81.94  573
ORG: precision:  85.18%; recall:  79.79%; FB1:  82.40  533
PER: precision:  87.85%; recall:  87.36%; FB1:  87.61  354




###############################
lang:  ar
precision    recall  f1-score   support

B-LOC       0.88      0.71      0.79      3780
I-ORG       0.76      0.87      0.81     10045
I-LOC       0.92      0.80      0.85      9073
I-PER       0.81      0.87      0.84      7937
B-ORG       0.76      0.76      0.76      3629
B-PER       0.82      0.82      0.82      3850

micro avg       0.82      0.82      0.82     38314
macro avg       0.82      0.81      0.81     38314
weighted avg       0.83      0.82      0.82     38314

processed 64347 tokens with 11259 phrases; found: 10564 phrases; correct: 8242.
accuracy:  82.24%; (non-O)
accuracy:  87.23%; precision:  78.02%; recall:  73.20%; FB1:  75.53
LOC: precision:  85.80%; recall:  69.23%; FB1:  76.63  3050
ORG: precision:  71.82%; recall:  72.28%; FB1:  72.05  3652
PER: precision:  77.73%; recall:  77.97%; FB1:  77.85  3862




###############################
lang:  bg
precision    recall  f1-score   support

B-LOC       0.89      0.91      0.90      6436
I-ORG       0.82      0.87      0.85      7964
I-LOC       0.85      0.78      0.82      3213
I-PER       0.89      0.88      0.89      4982
B-ORG       0.79      0.77      0.78      3670
B-PER       0.91      0.86      0.88      3954

micro avg       0.86      0.86      0.86     30219
macro avg       0.86      0.85      0.85     30219
weighted avg       0.86      0.86      0.86     30219

processed 83463 tokens with 14060 phrases; found: 13897 phrases; correct: 11836.
accuracy:  85.80%; (non-O)
accuracy:  93.83%; precision:  85.17%; recall:  84.18%; FB1:  84.67
LOC: precision:  88.08%; recall:  90.23%; FB1:  89.14  6593
ORG: precision:  75.27%; recall:  73.57%; FB1:  74.41  3587
PER: precision:  89.56%; recall:  84.19%; FB1:  86.79  3717




###############################
lang:  bn
precision    recall  f1-score   support

B-LOC       0.94      0.82      0.88       393
I-ORG       0.86      0.93      0.89      1031
I-LOC       0.94      0.82      0.88       703
I-PER       0.84      0.92      0.88       731
B-ORG       0.87      0.90      0.88       349
B-PER       0.87      0.91      0.89       347

micro avg       0.88      0.89      0.88      3554
macro avg       0.89      0.88      0.88      3554
weighted avg       0.88      0.89      0.88      3554

processed 4377 tokens with 1089 phrases; found: 1071 phrases; correct: 932.
accuracy:  89.00%; (non-O)
accuracy:  89.10%; precision:  87.02%; recall:  85.58%; FB1:  86.30
LOC: precision:  92.17%; recall:  80.92%; FB1:  86.18  345
ORG: precision:  85.40%; recall:  88.83%; FB1:  87.08  363
PER: precision:  83.75%; recall:  87.61%; FB1:  85.63  363




###############################
lang:  de
precision    recall  f1-score   support

B-LOC       0.78      0.77      0.78      4961
I-ORG       0.77      0.76      0.76      6043
I-LOC       0.77      0.58      0.66      2289
I-PER       0.96      0.84      0.89      6792
B-ORG       0.69      0.73      0.71      4157
B-PER       0.96      0.83      0.89      4750

micro avg       0.83      0.77      0.80     28992
macro avg       0.82      0.75      0.78     28992
weighted avg       0.84      0.77      0.80     28992

processed 97646 tokens with 13868 phrases; found: 13393 phrases; correct: 10307.
accuracy:  77.18%; (non-O)
accuracy:  91.95%; precision:  76.96%; recall:  74.32%; FB1:  75.62
LOC: precision:  75.34%; recall:  73.90%; FB1:  74.61  4866
ORG: precision:  64.23%; recall:  68.15%; FB1:  66.13  4411
PER: precision:  92.52%; recall:  80.17%; FB1:  85.90  4116




###############################
lang:  el
precision    recall  f1-score   support

B-LOC       0.84      0.84      0.84      4476
I-ORG       0.79      0.88      0.83      6685
I-LOC       0.74      0.54      0.62      1919
I-PER       0.90      0.87      0.88      5392
B-ORG       0.78      0.81      0.79      3655
B-PER       0.89      0.84      0.87      4032

micro avg       0.83      0.83      0.83     26159
macro avg       0.82      0.80      0.81     26159
weighted avg       0.83      0.83      0.83     26159

processed 90666 tokens with 12164 phrases; found: 12083 phrases; correct: 9880.
accuracy:  82.97%; (non-O)
accuracy:  94.09%; precision:  81.77%; recall:  81.22%; FB1:  81.49
LOC: precision:  82.55%; recall:  82.73%; FB1:  82.64  4486
ORG: precision:  75.29%; recall:  77.95%; FB1:  76.60  3784
PER: precision:  87.28%; recall:  82.52%; FB1:  84.83  3813




###############################
lang:  en
precision    recall  f1-score   support

B-LOC       0.80      0.77      0.78      4657
I-ORG       0.77      0.68      0.72     11607
I-LOC       0.87      0.62      0.72      6447
I-PER       0.93      0.75      0.83      7480
B-ORG       0.75      0.65      0.69      4745
B-PER       0.94      0.82      0.87      4556

micro avg       0.83      0.71      0.77     39492
macro avg       0.84      0.71      0.77     39492
weighted avg       0.84      0.71      0.76     39492

processed 80326 tokens with 13958 phrases; found: 12542 phrases; correct: 9604.
accuracy:  70.66%; (non-O)
accuracy:  84.66%; precision:  76.57%; recall:  68.81%; FB1:  72.48
LOC: precision:  72.53%; recall:  69.47%; FB1:  70.97  4460
ORG: precision:  67.03%; recall:  58.02%; FB1:  62.20  4107
PER: precision:  90.97%; recall:  79.37%; FB1:  84.77  3975




###############################
lang:  es
precision    recall  f1-score   support

B-LOC       0.94      0.85      0.89      4725
I-ORG       0.84      0.91      0.87     11371
I-LOC       0.90      0.73      0.81      6601
I-PER       0.95      0.86      0.91      7004
B-ORG       0.80      0.89      0.84      3576
B-PER       0.96      0.88      0.92      3959

micro avg       0.89      0.86      0.87     37236
macro avg       0.90      0.85      0.87     37236
weighted avg       0.89      0.86      0.87     37236

processed 64727 tokens with 12260 phrases; found: 11855 phrases; correct: 10412.
accuracy:  85.65%; (non-O)
accuracy:  91.26%; precision:  87.83%; recall:  84.93%; FB1:  86.35
LOC: precision:  91.20%; recall:  82.29%; FB1:  86.52  4263
ORG: precision:  78.06%; recall:  86.24%; FB1:  81.94  3951
PER: precision:  94.48%; recall:  86.89%; FB1:  90.53  3641




###############################
lang:  et
precision    recall  f1-score   support

B-LOC       0.82      0.82      0.82      5888
I-ORG       0.85      0.76      0.80      5731
I-LOC       0.71      0.73      0.72      2467
I-PER       0.95      0.86      0.90      5471
B-ORG       0.82      0.70      0.75      3875
B-PER       0.96      0.85      0.90      4129

micro avg       0.86      0.79      0.83     27561
macro avg       0.85      0.79      0.82     27561
weighted avg       0.86      0.79      0.83     27561

processed 80485 tokens with 13892 phrases; found: 12865 phrases; correct: 10397.
accuracy:  79.45%; (non-O)
accuracy:  91.98%; precision:  80.82%; recall:  74.84%; FB1:  77.71
LOC: precision:  75.94%; recall:  75.75%; FB1:  75.84  5873
ORG: precision:  75.55%; recall:  64.83%; FB1:  69.78  3325
PER: precision:  93.40%; recall:  82.95%; FB1:  87.87  3667




###############################
lang:  eu
precision    recall  f1-score   support

B-LOC       0.84      0.88      0.86      5682
I-ORG       0.86      0.75      0.80      5560
I-LOC       0.75      0.78      0.77      2876
I-PER       0.95      0.88      0.91      5449
B-ORG       0.81      0.72      0.77      3669
B-PER       0.94      0.83      0.88      4108

micro avg       0.87      0.82      0.84     27344
macro avg       0.86      0.81      0.83     27344
weighted avg       0.87      0.82      0.84     27344

processed 90661 tokens with 13459 phrases; found: 12843 phrases; correct: 10619.
accuracy:  81.56%; (non-O)
accuracy:  93.91%; precision:  82.68%; recall:  78.90%; FB1:  80.75
LOC: precision:  80.69%; recall:  84.72%; FB1:  82.66  5966
ORG: precision:  76.70%; recall:  68.08%; FB1:  72.13  3257
PER: precision:  91.35%; recall:  80.50%; FB1:  85.58  3620




###############################
lang:  fa
precision    recall  f1-score   support

B-LOC       0.91      0.81      0.86      3663
I-ORG       0.89      0.92      0.91     13255
I-LOC       0.92      0.85      0.88      8547
I-PER       0.85      0.87      0.86      7900
B-ORG       0.85      0.84      0.85      3535
B-PER       0.84      0.84      0.84      3544

micro avg       0.88      0.87      0.88     40444
macro avg       0.88      0.86      0.87     40444
weighted avg       0.88      0.87      0.88     40444

processed 59491 tokens with 10742 phrases; found: 10313 phrases; correct: 8793.
accuracy:  87.31%; (non-O)
accuracy:  90.55%; precision:  85.26%; recall:  81.86%; FB1:  83.52
LOC: precision:  89.56%; recall:  79.88%; FB1:  84.44  3267
ORG: precision:  83.85%; recall:  83.14%; FB1:  83.49  3505
PER: precision:  82.69%; recall:  82.62%; FB1:  82.65  3541




###############################
lang:  fi
precision    recall  f1-score   support

B-LOC       0.82      0.84      0.83      5629
I-ORG       0.84      0.79      0.81      5522
I-LOC       0.56      0.56      0.56      1096
I-PER       0.97      0.89      0.93      5437
B-ORG       0.79      0.69      0.74      4180
B-PER       0.97      0.88      0.92      4745

micro avg       0.86      0.81      0.84     26609
macro avg       0.83      0.78      0.80     26609
weighted avg       0.87      0.81      0.84     26609

processed 83660 tokens with 14554 phrases; found: 13685 phrases; correct: 11218.
accuracy:  81.22%; (non-O)
accuracy:  93.00%; precision:  81.97%; recall:  77.08%; FB1:  79.45
LOC: precision:  77.73%; recall:  79.13%; FB1:  78.42  5730
ORG: precision:  72.27%; recall:  63.47%; FB1:  67.58  3671
PER: precision:  95.96%; recall:  86.64%; FB1:  91.06  4284




###############################
lang:  fr
precision    recall  f1-score   support

B-LOC       0.90      0.79      0.84      4985
I-ORG       0.82      0.85      0.83     10386
I-LOC       0.89      0.72      0.79      5859
I-PER       0.95      0.81      0.87      6528
B-ORG       0.80      0.83      0.81      3885
B-PER       0.96      0.85      0.90      4499

micro avg       0.88      0.81      0.84     36142
macro avg       0.89      0.81      0.84     36142
weighted avg       0.88      0.81      0.84     36142

processed 68754 tokens with 13369 phrases; found: 12405 phrases; correct: 10621.
accuracy:  81.03%; (non-O)
accuracy:  89.43%; precision:  85.62%; recall:  79.44%; FB1:  82.42
LOC: precision:  86.39%; recall:  76.05%; FB1:  80.89  4388
ORG: precision:  75.96%; recall:  78.74%; FB1:  77.33  4027
PER: precision:  94.51%; recall:  83.82%; FB1:  88.84  3990




###############################
lang:  he
precision    recall  f1-score   support

B-LOC       0.81      0.67      0.73      5160
I-ORG       0.67      0.71      0.69      6907
I-LOC       0.73      0.56      0.63      3133
I-PER       0.76      0.83      0.79      6816
B-ORG       0.69      0.59      0.64      4142
B-PER       0.75      0.78      0.77      4396

micro avg       0.73      0.71      0.72     30554
macro avg       0.74      0.69      0.71     30554
weighted avg       0.74      0.71      0.72     30554

processed 85422 tokens with 13698 phrases; found: 12333 phrases; correct: 8741.
accuracy:  70.75%; (non-O)
accuracy:  87.43%; precision:  70.87%; recall:  63.81%; FB1:  67.16
LOC: precision:  78.18%; recall:  64.36%; FB1:  70.60  4248
ORG: precision:  62.20%; recall:  53.31%; FB1:  57.41  3550
PER: precision:  70.83%; recall:  73.07%; FB1:  71.93  4535




###############################
lang:  hi
precision    recall  f1-score   support

B-LOC       0.84      0.71      0.77       414
I-ORG       0.79      0.84      0.81      1123
I-LOC       0.80      0.55      0.65       398
I-PER       0.74      0.83      0.78       598
B-ORG       0.76      0.79      0.77       364
B-PER       0.82      0.82      0.82       450

micro avg       0.79      0.78      0.78      3347
macro avg       0.79      0.76      0.77      3347
weighted avg       0.79      0.78      0.78      3347

processed 6005 tokens with 1228 phrases; found: 1183 phrases; correct: 900.
accuracy:  77.92%; (non-O)
accuracy:  85.15%; precision:  76.08%; recall:  73.29%; FB1:  74.66
LOC: precision:  79.60%; recall:  67.87%; FB1:  73.27  353
ORG: precision:  72.11%; recall:  75.27%; FB1:  73.66  380
PER: precision:  76.67%; recall:  76.67%; FB1:  76.67  450




###############################
lang:  hu
precision    recall  f1-score   support

B-LOC       0.84      0.86      0.85      5671
I-ORG       0.81      0.82      0.81      5341
I-LOC       0.78      0.73      0.75      2404
I-PER       0.96      0.87      0.91      5501
B-ORG       0.81      0.77      0.79      3982
B-PER       0.96      0.86      0.91      4510

micro avg       0.87      0.83      0.85     27409
macro avg       0.86      0.82      0.84     27409
weighted avg       0.87      0.83      0.85     27409

processed 90302 tokens with 14163 phrases; found: 13631 phrases; correct: 11348.
accuracy:  82.81%; (non-O)
accuracy:  93.83%; precision:  83.25%; recall:  80.12%; FB1:  81.66
LOC: precision:  80.91%; recall:  83.09%; FB1:  81.98  5824
ORG: precision:  75.76%; recall:  71.82%; FB1:  73.74  3775
PER: precision:  93.65%; recall:  83.73%; FB1:  88.41  4032




###############################
lang:  id
precision    recall  f1-score   support

B-LOC       0.92      0.91      0.92      3745
I-ORG       0.87      0.90      0.89      8584
I-LOC       0.95      0.94      0.94      7809
I-PER       0.96      0.85      0.90      6520
B-ORG       0.86      0.87      0.87      3733
B-PER       0.96      0.86      0.91      3969

micro avg       0.92      0.89      0.91     34360
macro avg       0.92      0.89      0.90     34360
weighted avg       0.92      0.89      0.91     34360

processed 61834 tokens with 11447 phrases; found: 11094 phrases; correct: 9903.
accuracy:  89.21%; (non-O)
accuracy:  93.41%; precision:  89.26%; recall:  86.51%; FB1:  87.87
LOC: precision:  90.17%; recall:  89.88%; FB1:  90.02  3733
ORG: precision:  83.39%; recall:  84.89%; FB1:  84.14  3800
PER: precision:  94.58%; recall:  84.86%; FB1:  89.46  3561




###############################
lang:  it
precision    recall  f1-score   support

B-LOC       0.91      0.77      0.83      4820
I-ORG       0.83      0.82      0.83      9222
I-LOC       0.85      0.62      0.72      4366
I-PER       0.96      0.87      0.92      5794
B-ORG       0.80      0.81      0.81      4087
B-PER       0.97      0.90      0.93      4842

micro avg       0.88      0.81      0.84     33131
macro avg       0.89      0.80      0.84     33131
weighted avg       0.88      0.81      0.84     33131

processed 80871 tokens with 13749 phrases; found: 12679 phrases; correct: 10917.
accuracy:  80.69%; (non-O)
accuracy:  91.42%; precision:  86.10%; recall:  79.40%; FB1:  82.62
LOC: precision:  86.40%; recall:  73.05%; FB1:  79.17  4075
ORG: precision:  75.26%; recall:  76.29%; FB1:  75.77  4143
PER: precision:  95.90%; recall:  88.35%; FB1:  91.97  4461




###############################
lang:  ja
precision    recall  f1-score   support

B-LOC       0.83      0.51      0.64      5094
I-ORG       0.56      0.56      0.56     24814
I-LOC       0.83      0.50      0.62     17278
I-PER       0.84      0.50      0.63     21756
B-ORG       0.55      0.54      0.55      4267
B-PER       0.80      0.55      0.65      4085

micro avg       0.69      0.52      0.60     77294
macro avg       0.73      0.53      0.61     77294
weighted avg       0.73      0.52      0.60     77294

processed 306959 tokens with 13976 phrases; found: 10196 phrases; correct: 6870.
accuracy:  52.43%; (non-O)
accuracy:  86.02%; precision:  67.38%; recall:  49.16%; FB1:  56.84
LOC: precision:  80.60%; recall:  49.01%; FB1:  60.96  3134
ORG: precision:  52.36%; recall:  47.55%; FB1:  49.84  4236
PER: precision:  75.23%; recall:  51.14%; FB1:  60.89  2826




###############################
lang:  jv
precision    recall  f1-score   support

B-LOC       0.88      0.85      0.86        52
I-ORG       0.71      0.76      0.74        66
I-LOC       0.67      0.70      0.68        43
I-PER       0.84      0.70      0.77        44
B-ORG       0.74      0.78      0.76        40
B-PER       0.90      0.72      0.80        25

micro avg       0.77      0.76      0.76       270
macro avg       0.79      0.75      0.77       270
weighted avg       0.78      0.76      0.77       270

processed 678 tokens with 117 phrases; found: 112 phrases; correct: 90.
accuracy:  75.56%; (non-O)
accuracy:  88.94%; precision:  80.36%; recall:  76.92%; FB1:  78.60
LOC: precision:  84.00%; recall:  80.77%; FB1:  82.35  50
ORG: precision:  73.81%; recall:  77.50%; FB1:  75.61  42
PER: precision:  85.00%; recall:  68.00%; FB1:  75.56  20




###############################
lang:  ka
precision    recall  f1-score   support

B-LOC       0.82      0.72      0.77      5288
I-ORG       0.84      0.83      0.83      7800
I-LOC       0.72      0.59      0.65      2191
I-PER       0.80      0.88      0.84      4666
B-ORG       0.79      0.66      0.72      3807
B-PER       0.80      0.81      0.80      3962

micro avg       0.81      0.77      0.79     27714
macro avg       0.79      0.75      0.77     27714
weighted avg       0.81      0.77      0.79     27714

processed 81921 tokens with 13057 phrases; found: 11833 phrases; correct: 9017.
accuracy:  77.30%; (non-O)
accuracy:  90.06%; precision:  76.20%; recall:  69.06%; FB1:  72.45
LOC: precision:  78.11%; recall:  68.68%; FB1:  73.09  4650
ORG: precision:  73.14%; recall:  61.15%; FB1:  66.61  3183
PER: precision:  76.42%; recall:  77.16%; FB1:  76.79  4000




###############################
lang:  kk
precision    recall  f1-score   support

B-LOC       0.76      0.79      0.77       383
I-ORG       0.63      0.80      0.70       592
I-LOC       0.78      0.49      0.60       210
I-PER       0.84      0.84      0.84       466
B-ORG       0.63      0.61      0.62       355
B-PER       0.82      0.74      0.78       377

micro avg       0.72      0.74      0.73      2383
macro avg       0.74      0.71      0.72      2383
weighted avg       0.73      0.74      0.73      2383

processed 7936 tokens with 1115 phrases; found: 1081 phrases; correct: 744.
accuracy:  74.11%; (non-O)
accuracy:  89.57%; precision:  68.83%; recall:  66.73%; FB1:  67.76
LOC: precision:  74.62%; recall:  77.55%; FB1:  76.06  398
ORG: precision:  52.48%; recall:  50.70%; FB1:  51.58  343
PER: precision:  78.53%; recall:  70.82%; FB1:  74.48  340




###############################
lang:  ko
precision    recall  f1-score   support

B-LOC       0.88      0.81      0.85      5855
I-ORG       0.75      0.81      0.78      5437
I-LOC       0.79      0.82      0.80      2712
I-PER       0.74      0.84      0.79      3468
B-ORG       0.81      0.66      0.72      4319
B-PER       0.76      0.80      0.78      4249

micro avg       0.79      0.79      0.79     26040
macro avg       0.79      0.79      0.79     26040
weighted avg       0.79      0.79      0.79     26040

processed 80841 tokens with 14423 phrases; found: 13369 phrases; correct: 10274.
accuracy:  78.94%; (non-O)
accuracy:  90.59%; precision:  76.85%; recall:  71.23%; FB1:  73.93
LOC: precision:  83.32%; recall:  77.23%; FB1:  80.16  5427
ORG: precision:  71.82%; recall:  58.35%; FB1:  64.38  3509
PER: precision:  72.91%; recall:  76.06%; FB1:  74.45  4433




###############################
lang:  ml
precision    recall  f1-score   support

B-LOC       0.86      0.60      0.70       443
I-ORG       0.80      0.88      0.83       774
I-LOC       0.80      0.32      0.46       219
I-PER       0.73      0.85      0.78       492
B-ORG       0.74      0.71      0.72       354
B-PER       0.72      0.80      0.76       407

micro avg       0.77      0.75      0.76      2689
macro avg       0.77      0.69      0.71      2689
weighted avg       0.78      0.75      0.75      2689

processed 6727 tokens with 1204 phrases; found: 1101 phrases; correct: 810.
accuracy:  74.64%; (non-O)
accuracy:  87.88%; precision:  73.57%; recall:  67.28%; FB1:  70.28
LOC: precision:  83.06%; recall:  57.56%; FB1:  68.00  307
ORG: precision:  70.06%; recall:  68.08%; FB1:  69.05  344
PER: precision:  69.78%; recall:  77.15%; FB1:  73.28  450




###############################
lang:  mr
precision    recall  f1-score   support

B-LOC       0.84      0.69      0.76       525
I-ORG       0.78      0.91      0.84       852
I-LOC       0.67      0.49      0.57       258
I-PER       0.85      0.79      0.82       598
B-ORG       0.78      0.74      0.76       364
B-PER       0.82      0.79      0.80       375

micro avg       0.80      0.78      0.79      2972
macro avg       0.79      0.74      0.76      2972
weighted avg       0.80      0.78      0.78      2972

processed 7356 tokens with 1264 phrases; found: 1142 phrases; correct: 900.
accuracy:  77.59%; (non-O)
accuracy:  88.93%; precision:  78.81%; recall:  71.20%; FB1:  74.81
LOC: precision:  81.57%; recall:  67.43%; FB1:  73.83  434
ORG: precision:  73.85%; recall:  70.60%; FB1:  72.19  348
PER: precision:  80.28%; recall:  77.07%; FB1:  78.64  360




###############################
lang:  ms
precision    recall  f1-score   support

B-LOC       0.92      0.94      0.93       367
I-ORG       0.86      0.89      0.87       913
I-LOC       0.97      0.95      0.96       898
I-PER       0.95      0.78      0.86       555
B-ORG       0.82      0.85      0.84       375
B-PER       0.95      0.84      0.89       373

micro avg       0.91      0.88      0.90      3481
macro avg       0.91      0.87      0.89      3481
weighted avg       0.91      0.88      0.90      3481

processed 5874 tokens with 1115 phrases; found: 1087 phrases; correct: 952.
accuracy:  88.28%; (non-O)
accuracy:  92.12%; precision:  87.58%; recall:  85.38%; FB1:  86.47
LOC: precision:  91.94%; recall:  93.19%; FB1:  92.56  372
ORG: precision:  78.81%; recall:  81.33%; FB1:  80.05  387
PER: precision:  92.99%; recall:  81.77%; FB1:  87.02  328




###############################
lang:  my
precision    recall  f1-score   support

B-LOC       0.55      0.38      0.45        56
I-ORG       0.63      0.49      0.55        68
I-LOC       0.50      0.75      0.60         4
I-PER       0.30      0.67      0.41        46
B-ORG       0.84      0.48      0.62        33
B-PER       0.31      0.53      0.40        30

micro avg       0.44      0.51      0.47       237
macro avg       0.52      0.55      0.50       237
weighted avg       0.54      0.51      0.49       237

processed 756 tokens with 119 phrases; found: 108 phrases; correct: 46.
accuracy:  50.63%; (non-O)
accuracy:  74.07%; precision:  42.59%; recall:  38.66%; FB1:  40.53
LOC: precision:  55.26%; recall:  37.50%; FB1:  44.68  38
ORG: precision:  68.42%; recall:  39.39%; FB1:  50.00  19
PER: precision:  23.53%; recall:  40.00%; FB1:  29.63  51




###############################
lang:  nl
precision    recall  f1-score   support

B-LOC       0.90      0.85      0.87      5133
I-ORG       0.83      0.79      0.81      6693
I-LOC       0.90      0.68      0.77      3662
I-PER       0.96      0.86      0.91      6371
B-ORG       0.82      0.80      0.81      3908
B-PER       0.96      0.88      0.92      4684

micro avg       0.89      0.82      0.85     30451
macro avg       0.89      0.81      0.85     30451
weighted avg       0.89      0.82      0.85     30451

processed 85122 tokens with 13725 phrases; found: 12947 phrases; correct: 11196.
accuracy:  81.69%; (non-O)
accuracy:  92.93%; precision:  86.48%; recall:  81.57%; FB1:  83.95
LOC: precision:  86.27%; recall:  81.01%; FB1:  83.55  4820
ORG: precision:  78.22%; recall:  76.92%; FB1:  77.56  3843
PER: precision:  94.12%; recall:  86.08%; FB1:  89.92  4284




###############################
lang:  pt
precision    recall  f1-score   support

B-LOC       0.92      0.85      0.89      4779
I-ORG       0.83      0.88      0.85     10542
I-LOC       0.89      0.71      0.79      6467
I-PER       0.96      0.81      0.88      7310
B-ORG       0.81      0.85      0.83      3753
B-PER       0.96      0.85      0.90      4291

micro avg       0.88      0.83      0.85     37142
macro avg       0.89      0.82      0.86     37142
weighted avg       0.89      0.83      0.85     37142

processed 63647 tokens with 12823 phrases; found: 12187 phrases; correct: 10475.
accuracy:  82.53%; (non-O)
accuracy:  89.27%; precision:  85.95%; recall:  81.69%; FB1:  83.77
LOC: precision:  87.47%; recall:  81.04%; FB1:  84.13  4428
ORG: precision:  77.42%; recall:  81.32%; FB1:  79.32  3942
PER: precision:  93.00%; recall:  82.73%; FB1:  87.57  3817




###############################
lang:  ru
precision    recall  f1-score   support

B-LOC       0.75      0.81      0.78      4560
I-ORG       0.78      0.83      0.80      8008
I-LOC       0.63      0.69      0.66      3060
I-PER       0.93      0.83      0.88      7544
B-ORG       0.78      0.72      0.75      4074
B-PER       0.90      0.79      0.84      3543

micro avg       0.80      0.79      0.80     30789
macro avg       0.79      0.78      0.78     30789
weighted avg       0.81      0.79      0.80     30789

processed 71288 tokens with 12177 phrases; found: 11798 phrases; correct: 9093.
accuracy:  79.34%; (non-O)
accuracy:  89.47%; precision:  77.07%; recall:  74.67%; FB1:  75.85
LOC: precision:  74.10%; recall:  79.36%; FB1:  76.64  4884
ORG: precision:  71.75%; recall:  66.40%; FB1:  68.97  3770
PER: precision:  88.07%; recall:  78.15%; FB1:  82.82  3144




###############################
lang:  sw
precision    recall  f1-score   support

B-LOC       0.86      0.84      0.85       388
I-ORG       0.79      0.85      0.82       763
I-LOC       0.80      0.67      0.73       568
I-PER       0.97      0.87      0.92       744
B-ORG       0.84      0.88      0.86       374
B-PER       0.97      0.88      0.92       432

micro avg       0.87      0.83      0.85      3269
macro avg       0.87      0.83      0.85      3269
weighted avg       0.87      0.83      0.85      3269

processed 5786 tokens with 1194 phrases; found: 1161 phrases; correct: 1003.
accuracy:  82.90%; (non-O)
accuracy:  89.63%; precision:  86.39%; recall:  84.00%; FB1:  85.18
LOC: precision:  81.33%; recall:  78.61%; FB1:  79.95  375
ORG: precision:  81.98%; recall:  86.36%; FB1:  84.11  394
PER: precision:  95.66%; recall:  86.81%; FB1:  91.02  392




###############################
lang:  ta
precision    recall  f1-score   support

B-LOC       0.81      0.71      0.75       436
I-ORG       0.77      0.83      0.80       814
I-LOC       0.75      0.51      0.61       239
I-PER       0.84      0.91      0.87       615
B-ORG       0.77      0.68      0.72       383
B-PER       0.77      0.85      0.81       422

micro avg       0.79      0.79      0.79      2909
macro avg       0.78      0.75      0.76      2909
weighted avg       0.79      0.79      0.78      2909

processed 7234 tokens with 1241 phrases; found: 1179 phrases; correct: 869.
accuracy:  78.51%; (non-O)
accuracy:  88.98%; precision:  73.71%; recall:  70.02%; FB1:  71.82
LOC: precision:  77.89%; recall:  67.89%; FB1:  72.55  380
ORG: precision:  69.44%; recall:  61.10%; FB1:  65.00  337
PER: precision:  73.38%; recall:  80.33%; FB1:  76.70  462




###############################
lang:  te
precision    recall  f1-score   support

B-LOC       0.76      0.51      0.61       450
I-ORG       0.70      0.74      0.72       633
I-LOC       0.71      0.44      0.55       178
I-PER       0.50      0.77      0.61       294
B-ORG       0.61      0.57      0.59       340
B-PER       0.55      0.67      0.61       381

micro avg       0.63      0.64      0.63      2276
macro avg       0.64      0.62      0.61      2276
weighted avg       0.65      0.64      0.63      2276

processed 8155 tokens with 1171 phrases; found: 1083 phrases; correct: 627.
accuracy:  63.75%; (non-O)
accuracy:  85.62%; precision:  57.89%; recall:  53.54%; FB1:  55.63
LOC: precision:  73.91%; recall:  49.11%; FB1:  59.01  299
ORG: precision:  53.89%; recall:  50.88%; FB1:  52.34  321
PER: precision:  50.32%; recall:  61.15%; FB1:  55.21  463




###############################
lang:  th
precision    recall  f1-score   support

B-LOC       0.85      0.55      0.67      6503
I-ORG       0.63      0.67      0.65     56831
I-LOC       0.84      0.55      0.66     47608
I-PER       0.86      0.65      0.74     57522
B-ORG       0.50      0.54      0.52      5151
B-PER       0.48      0.60      0.53      5316

micro avg       0.73      0.62      0.67    178931
macro avg       0.69      0.59      0.63    178931
weighted avg       0.76      0.62      0.67    178931

processed 649606 tokens with 20897 phrases; found: 16403 phrases; correct: 11059.
accuracy:  61.94%; (non-O)
accuracy:  87.63%; precision:  67.42%; recall:  52.92%; FB1:  59.30
LOC: precision:  80.18%; recall:  51.38%; FB1:  62.62  4238
ORG: precision:  48.03%; recall:  44.13%; FB1:  46.00  5469
PER: precision:  75.18%; recall:  60.43%; FB1:  67.00  6696




###############################
lang:  tl
precision    recall  f1-score   support

B-LOC       0.83      0.90      0.86       327
I-ORG       0.83      0.81      0.82      1045
I-LOC       0.87      0.85      0.86       706
I-PER       0.95      0.84      0.89       813
B-ORG       0.79      0.82      0.81       341
B-PER       0.94      0.86      0.90       366

micro avg       0.87      0.84      0.85      3598
macro avg       0.87      0.85      0.86      3598
weighted avg       0.87      0.84      0.85      3598

processed 4627 tokens with 1034 phrases; found: 1040 phrases; correct: 857.
accuracy:  83.82%; (non-O)
accuracy:  86.73%; precision:  82.40%; recall:  82.88%; FB1:  82.64
LOC: precision:  79.26%; recall:  85.32%; FB1:  82.18  352
ORG: precision:  76.49%; recall:  79.18%; FB1:  77.81  353
PER: precision:  91.94%; recall:  84.15%; FB1:  87.87  335




###############################
lang:  tr
precision    recall  f1-score   support

B-LOC       0.87      0.79      0.83      4914
I-ORG       0.78      0.89      0.83      6979
I-LOC       0.83      0.68      0.75      3005
I-PER       0.95      0.84      0.89      5694
B-ORG       0.78      0.82      0.80      4154
B-PER       0.95      0.84      0.89      4519

micro avg       0.85      0.82      0.84     29265
macro avg       0.86      0.81      0.83     29265
weighted avg       0.86      0.82      0.84     29265

processed 75731 tokens with 13587 phrases; found: 12822 phrases; correct: 10708.
accuracy:  82.40%; (non-O)
accuracy:  92.07%; precision:  83.51%; recall:  78.81%; FB1:  81.09
LOC: precision:  84.17%; recall:  76.82%; FB1:  80.33  4485
ORG: precision:  73.45%; recall:  77.25%; FB1:  75.30  4369
PER: precision:  93.85%; recall:  82.41%; FB1:  87.76  3968




###############################
lang:  ur
precision    recall  f1-score   support

B-LOC       0.92      0.87      0.90       334
I-ORG       0.86      0.89      0.88      1005
I-LOC       0.91      0.90      0.91       904
I-PER       0.89      0.91      0.90       928
B-ORG       0.86      0.86      0.86       323
B-PER       0.89      0.89      0.89       363

micro avg       0.89      0.89      0.89      3857
macro avg       0.89      0.89      0.89      3857
weighted avg       0.89      0.89      0.89      3857

processed 5027 tokens with 1020 phrases; found: 1003 phrases; correct: 878.
accuracy:  89.50%; (non-O)
accuracy:  90.89%; precision:  87.54%; recall:  86.08%; FB1:  86.80
LOC: precision:  90.54%; recall:  85.93%; FB1:  88.17  317
ORG: precision:  86.07%; recall:  86.07%; FB1:  86.07  323
PER: precision:  86.23%; recall:  86.23%; FB1:  86.23  363




###############################
lang:  vi
precision    recall  f1-score   support

B-LOC       0.88      0.86      0.87      3717
I-ORG       0.86      0.86      0.86     13562
I-LOC       0.89      0.85      0.87      8018
I-PER       0.93      0.81      0.87      7787
B-ORG       0.82      0.82      0.82      3704
B-PER       0.92      0.85      0.88      3884

micro avg       0.88      0.84      0.86     40672
macro avg       0.88      0.84      0.86     40672
weighted avg       0.88      0.84      0.86     40672

processed 64967 tokens with 11305 phrases; found: 10904 phrases; correct: 9223.
accuracy:  84.22%; (non-O)
accuracy:  89.37%; precision:  84.58%; recall:  81.58%; FB1:  83.06
LOC: precision:  85.10%; recall:  83.72%; FB1:  84.40  3657
ORG: precision:  78.77%; recall:  78.35%; FB1:  78.56  3684
PER: precision:  90.06%; recall:  82.62%; FB1:  86.18  3563




###############################
lang:  yo
precision    recall  f1-score   support

B-LOC       0.73      0.77      0.75        39
I-ORG       0.76      0.79      0.78        87
I-LOC       0.90      0.89      0.90        72
I-PER       0.94      0.72      0.82        71
B-ORG       0.68      0.79      0.73        29
B-PER       0.97      0.70      0.81        43

micro avg       0.83      0.78      0.81       341
macro avg       0.83      0.78      0.80       341
weighted avg       0.84      0.78      0.81       341

processed 503 tokens with 111 phrases; found: 106 phrases; correct: 81.
accuracy:  78.30%; (non-O)
accuracy:  85.09%; precision:  76.42%; recall:  72.97%; FB1:  74.65
LOC: precision:  73.17%; recall:  76.92%; FB1:  75.00  41
ORG: precision:  64.71%; recall:  75.86%; FB1:  69.84  34
PER: precision:  93.55%; recall:  67.44%; FB1:  78.38  31




###############################
lang:  zh
precision    recall  f1-score   support

B-LOC       0.83      0.71      0.76      4371
I-ORG       0.75      0.68      0.71     17399
I-LOC       0.85      0.74      0.79     12282
I-PER       0.87      0.77      0.82     12897
B-ORG       0.67      0.65      0.66      3779
B-PER       0.81      0.76      0.79      3899

micro avg       0.80      0.72      0.76     54627
macro avg       0.80      0.72      0.75     54627
weighted avg       0.80      0.72      0.76     54627

processed 207505 tokens with 12532 phrases; found: 11033 phrases; correct: 8345.
accuracy:  72.07%; (non-O)
accuracy:  90.67%; precision:  75.64%; recall:  66.59%; FB1:  70.83
LOC: precision:  80.35%; recall:  66.99%; FB1:  73.06  3730
ORG: precision:  67.63%; recall:  59.85%; FB1:  63.50  3642
PER: precision:  78.80%; recall:  73.17%; FB1:  75.88  3661
```
