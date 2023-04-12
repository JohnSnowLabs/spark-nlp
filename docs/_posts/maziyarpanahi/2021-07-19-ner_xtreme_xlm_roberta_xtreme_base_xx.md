---
layout: model
title: Detect Entities in 40 languages - XTREME (ner_xtreme_xlm_roberta_xtreme_base)
author: John Snow Labs
name: ner_xtreme_xlm_roberta_xtreme_base
date: 2021-07-19
tags: [open_source, xx, multilingual, ner, xtreme, xlm_roberta]
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

XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization. This NER model was trained over the XTREME dataset by using XlmRoBertaEmbeddings (xlm_roberta_xtreme_base).

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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_xtreme_xlm_roberta_xtreme_base_xx_3.1.3_2.4_1626711340421.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ner_xtreme_xlm_roberta_xtreme_base_xx_3.1.3_2.4_1626711340421.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = XlmRoBertaEmbeddings\
    .pretrained('xlm_roberta_xtreme_base', 'xx')\
    .setInputCols(["token", "document"])\
    .setOutputCol("embeddings")

ner_model = NerDLModel.pretrained('ner_xtreme_xlm_roberta_xtreme_base', 'xx') \
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

val embeddings = XlmRoBertaEmbeddings.pretrained("xlm_roberta_xtreme_base", "xx")
    .setInputCols(Array("document", "token"))
    .setOutputCol("embeddings")

val ner_model = NerDLModel.pretrained("ner_xtreme_xlm_roberta_xtreme_base", "xx") 
    .setInputCols(Array("document", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter() 
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

ner_df = nlu.load('xx.ner.ner_xtreme_xlm_roberta_xtreme_base').predict(text, output_level='token')
```
</div>

## Results
```bash
+-----------------+---------+
|chunk            |ner_label|
+-----------------+---------+
|Jerome Horsey    |PER      |
|Russia Company   |ORG      |
|Moscow           |LOC      |
|Emilie Hartmanns |PER      |
|August Hartmann  |PER      |
|Hohen Karlsschule|ORG      |
|Stuttgart        |LOC      |
|James Watt       |PER      |
|Scozia           |LOC      |
|John             |PER      |
|Alaska           |LOC      |
|Starbucks        |ORG      |
+-----------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_xtreme_xlm_roberta_xtreme_base|
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
Average of all languages benchmark (multi-label classification and CoNLL Eval):


precision    recall  f1-score   support

B-LOC       0.87      0.89      0.88    129861
I-ORG       0.82      0.84      0.83    291145
I-MISC       0.00      0.00      0.00         0
I-LOC       0.81      0.84      0.83    179310
I-PER       0.87      0.89      0.88    234076
B-MISC       0.00      0.00      0.00         0
B-ORG       0.85      0.80      0.82    105547
B-PER       0.91      0.90      0.91    114118

micro avg       0.85      0.86      0.85   1054057
macro avg       0.64      0.64      0.64   1054057
weighted avg       0.85      0.86      0.85   1054057

processed 2928018 tokens with 349526 phrases; found: 344025 phrases; correct: 292983.
accuracy:  85.78%; (non-O)
accuracy:  92.66%; precision:  85.16%; recall:  83.82%; FB1:  84.49
LOC: precision:  84.65%; recall:  86.65%; FB1:  85.64  132937
ORG: precision:  81.27%; recall:  76.33%; FB1:  78.72  99127
PER: precision:  89.22%; recall:  87.53%; FB1:  88.37  111961


###############################


Language by language benchmarks (multi-label classification and CoNLL Eval):


lang:  af
precision    recall  f1-score   support

B-LOC       0.84      0.92      0.88       562
I-ORG       0.92      0.93      0.92       786
I-LOC       0.70      0.80      0.74       198
I-PER       0.94      0.97      0.95       504
B-ORG       0.92      0.84      0.88       569
B-PER       0.92      0.96      0.94       356

micro avg       0.89      0.91      0.90      2975
macro avg       0.87      0.90      0.89      2975
weighted avg       0.89      0.91      0.90      2975

processed 10808 tokens with 1487 phrases; found: 1506 phrases; correct: 1323.
accuracy:  91.29%; (non-O)
accuracy:  96.50%; precision:  87.85%; recall:  88.97%; FB1:  88.41
LOC: precision:  82.52%; recall:  90.75%; FB1:  86.44  618
ORG: precision:  91.52%; recall:  83.48%; FB1:  87.32  519
PER: precision:  91.60%; recall:  94.94%; FB1:  93.24  369




###############################
lang:  ar
precision    recall  f1-score   support

B-LOC       0.87      0.90      0.88      3780
I-ORG       0.89      0.89      0.89     10045
I-LOC       0.90      0.93      0.92      9073
I-PER       0.90      0.89      0.89      7937
B-ORG       0.89      0.82      0.85      3629
B-PER       0.88      0.88      0.88      3850

micro avg       0.89      0.89      0.89     38314
macro avg       0.89      0.88      0.89     38314
weighted avg       0.89      0.89      0.89     38314

processed 64347 tokens with 11259 phrases; found: 11109 phrases; correct: 9447.
accuracy:  89.22%; (non-O)
accuracy:  92.67%; precision:  85.04%; recall:  83.91%; FB1:  84.47
LOC: precision:  85.04%; recall:  88.15%; FB1:  86.57  3918
ORG: precision:  85.00%; recall:  78.86%; FB1:  81.82  3367
PER: precision:  85.07%; recall:  84.49%; FB1:  84.78  3824




###############################
lang:  bg
precision    recall  f1-score   support

B-LOC       0.92      0.95      0.94      6436
I-ORG       0.91      0.89      0.90      7964
I-LOC       0.85      0.89      0.87      3213
I-PER       0.91      0.94      0.93      4982
B-ORG       0.88      0.82      0.85      3670
B-PER       0.92      0.94      0.93      3954

micro avg       0.91      0.91      0.91     30219
macro avg       0.90      0.91      0.90     30219
weighted avg       0.91      0.91      0.91     30219

processed 83463 tokens with 14060 phrases; found: 14076 phrases; correct: 12687.
accuracy:  91.03%; (non-O)
accuracy:  95.94%; precision:  90.13%; recall:  90.23%; FB1:  90.18
LOC: precision:  91.61%; recall:  94.33%; FB1:  92.95  6627
ORG: precision:  85.89%; recall:  79.81%; FB1:  82.74  3410
PER: precision:  91.28%; recall:  93.25%; FB1:  92.26  4039




###############################
lang:  bn
precision    recall  f1-score   support

B-LOC       0.85      0.93      0.89       393
I-ORG       0.93      0.91      0.92      1031
I-LOC       0.86      0.91      0.89       703
I-PER       0.95      0.93      0.94       731
B-ORG       0.92      0.90      0.91       349
B-PER       0.95      0.92      0.94       347

micro avg       0.91      0.92      0.91      3554
macro avg       0.91      0.92      0.91      3554
weighted avg       0.92      0.92      0.91      3554

processed 4377 tokens with 1089 phrases; found: 1100 phrases; correct: 979.
accuracy:  91.50%; (non-O)
accuracy:  92.28%; precision:  89.00%; recall:  89.90%; FB1:  89.45
LOC: precision:  84.04%; recall:  91.09%; FB1:  87.42  426
ORG: precision:  90.56%; recall:  87.97%; FB1:  89.24  339
PER: precision:  93.73%; recall:  90.49%; FB1:  92.08  335




###############################
lang:  de
precision    recall  f1-score   support

B-LOC       0.86      0.89      0.87      4961
I-ORG       0.88      0.87      0.87      6043
I-LOC       0.80      0.80      0.80      2289
I-PER       0.96      0.94      0.95      6792
B-ORG       0.82      0.79      0.81      4157
B-PER       0.95      0.92      0.94      4750

micro avg       0.89      0.88      0.89     28992
macro avg       0.88      0.87      0.87     28992
weighted avg       0.89      0.88      0.89     28992

processed 97646 tokens with 13868 phrases; found: 13738 phrases; correct: 11809.
accuracy:  88.27%; (non-O)
accuracy:  95.84%; precision:  85.96%; recall:  85.15%; FB1:  85.55
LOC: precision:  84.20%; recall:  87.32%; FB1:  85.73  5145
ORG: precision:  79.33%; recall:  76.52%; FB1:  77.90  4010
PER: precision:  93.74%; recall:  90.44%; FB1:  92.06  4583




###############################
lang:  el
precision    recall  f1-score   support

B-LOC       0.88      0.91      0.89      4476
I-ORG       0.89      0.88      0.89      6685
I-LOC       0.72      0.76      0.74      1919
I-PER       0.91      0.94      0.92      5392
B-ORG       0.88      0.83      0.86      3655
B-PER       0.91      0.93      0.92      4032

micro avg       0.88      0.89      0.88     26159
macro avg       0.86      0.88      0.87     26159
weighted avg       0.88      0.89      0.88     26159

processed 90666 tokens with 12164 phrases; found: 12254 phrases; correct: 10675.
accuracy:  89.03%; (non-O)
accuracy:  95.89%; precision:  87.11%; recall:  87.76%; FB1:  87.44
LOC: precision:  86.10%; recall:  89.57%; FB1:  87.80  4656
ORG: precision:  86.00%; recall:  81.34%; FB1:  83.61  3457
PER: precision:  89.18%; recall:  91.57%; FB1:  90.36  4141




###############################
lang:  en
precision    recall  f1-score   support

B-LOC       0.82      0.87      0.84      4657
I-ORG       0.83      0.85      0.84     11607
I-LOC       0.86      0.73      0.79      6447
I-PER       0.89      0.88      0.88      7480
B-ORG       0.82      0.77      0.79      4745
B-PER       0.90      0.91      0.91      4556

micro avg       0.85      0.84      0.84     39492
macro avg       0.85      0.83      0.84     39492
weighted avg       0.85      0.84      0.84     39492

processed 80326 tokens with 13958 phrases; found: 13975 phrases; correct: 11183.
accuracy:  83.53%; (non-O)
accuracy:  90.98%; precision:  80.02%; recall:  80.12%; FB1:  80.07
LOC: precision:  75.34%; recall:  79.30%; FB1:  77.27  4902
ORG: precision:  76.77%; recall:  71.80%; FB1:  74.20  4438
PER: precision:  88.09%; recall:  89.62%; FB1:  88.85  4635




###############################
lang:  es
precision    recall  f1-score   support

B-LOC       0.92      0.92      0.92      4725
I-ORG       0.89      0.92      0.91     11371
I-LOC       0.86      0.86      0.86      6601
I-PER       0.95      0.91      0.93      7004
B-ORG       0.88      0.88      0.88      3576
B-PER       0.95      0.93      0.94      3959

micro avg       0.91      0.91      0.91     37236
macro avg       0.91      0.90      0.91     37236
weighted avg       0.91      0.91      0.91     37236

processed 64727 tokens with 12260 phrases; found: 12210 phrases; correct: 11032.
accuracy:  90.55%; (non-O)
accuracy:  94.06%; precision:  90.35%; recall:  89.98%; FB1:  90.17
LOC: precision:  90.16%; recall:  90.73%; FB1:  90.44  4755
ORG: precision:  86.45%; recall:  86.35%; FB1:  86.40  3572
PER: precision:  94.18%; recall:  92.37%; FB1:  93.27  3883




###############################
lang:  et
precision    recall  f1-score   support

B-LOC       0.91      0.94      0.92      5888
I-ORG       0.90      0.88      0.89      5731
I-LOC       0.84      0.85      0.85      2467
I-PER       0.96      0.94      0.95      5471
B-ORG       0.89      0.82      0.86      3875
B-PER       0.95      0.95      0.95      4129

micro avg       0.92      0.90      0.91     27561
macro avg       0.91      0.90      0.90     27561
weighted avg       0.92      0.90      0.91     27561

processed 80485 tokens with 13892 phrases; found: 13760 phrases; correct: 12281.
accuracy:  90.48%; (non-O)
accuracy:  96.05%; precision:  89.25%; recall:  88.40%; FB1:  88.83
LOC: precision:  88.42%; recall:  91.53%; FB1:  89.94  6095
ORG: precision:  85.56%; recall:  78.89%; FB1:  82.09  3573
PER: precision:  93.72%; recall:  92.88%; FB1:  93.30  4092




###############################
lang:  eu
precision    recall  f1-score   support

B-LOC       0.91      0.94      0.93      5682
I-ORG       0.91      0.84      0.87      5560
I-LOC       0.79      0.89      0.84      2876
I-PER       0.95      0.94      0.94      5449
B-ORG       0.91      0.81      0.86      3669
B-PER       0.94      0.93      0.93      4108

micro avg       0.91      0.90      0.90     27344
macro avg       0.90      0.89      0.90     27344
weighted avg       0.91      0.90      0.90     27344

processed 90661 tokens with 13459 phrases; found: 13219 phrases; correct: 11812.
accuracy:  89.68%; (non-O)
accuracy:  96.37%; precision:  89.36%; recall:  87.76%; FB1:  88.55
LOC: precision:  88.89%; recall:  91.99%; FB1:  90.42  5880
ORG: precision:  87.56%; recall:  78.30%; FB1:  82.68  3281
PER: precision:  91.47%; recall:  90.36%; FB1:  90.91  4058




###############################
lang:  fa
precision    recall  f1-score   support

B-LOC       0.91      0.92      0.92      3663
I-ORG       0.94      0.96      0.95     13255
I-LOC       0.92      0.93      0.92      8547
I-PER       0.94      0.92      0.93      7900
B-ORG       0.91      0.91      0.91      3535
B-PER       0.93      0.91      0.92      3544

micro avg       0.93      0.93      0.93     40444
macro avg       0.93      0.92      0.92     40444
weighted avg       0.93      0.93      0.93     40444

processed 59491 tokens with 10742 phrases; found: 10702 phrases; correct: 9699.
accuracy:  93.10%; (non-O)
accuracy:  94.77%; precision:  90.63%; recall:  90.29%; FB1:  90.46
LOC: precision:  89.42%; recall:  90.66%; FB1:  90.04  3714
ORG: precision:  90.29%; recall:  89.99%; FB1:  90.14  3523
PER: precision:  92.27%; recall:  90.21%; FB1:  91.23  3465




###############################
lang:  fi
precision    recall  f1-score   support

B-LOC       0.89      0.92      0.90      5629
I-ORG       0.90      0.89      0.90      5522
I-LOC       0.69      0.75      0.72      1096
I-PER       0.96      0.96      0.96      5437
B-ORG       0.88      0.82      0.85      4180
B-PER       0.95      0.95      0.95      4745

micro avg       0.91      0.90      0.91     26609
macro avg       0.88      0.88      0.88     26609
weighted avg       0.91      0.90      0.91     26609

processed 83660 tokens with 14554 phrases; found: 14403 phrases; correct: 12760.
accuracy:  90.35%; (non-O)
accuracy:  96.03%; precision:  88.59%; recall:  87.67%; FB1:  88.13
LOC: precision:  86.57%; recall:  88.99%; FB1:  87.76  5786
ORG: precision:  84.51%; recall:  78.47%; FB1:  81.38  3881
PER: precision:  94.40%; recall:  94.23%; FB1:  94.31  4736




###############################
lang:  fr
precision    recall  f1-score   support

B-LOC       0.90      0.89      0.89      4985
I-ORG       0.87      0.91      0.89     10386
I-LOC       0.84      0.85      0.85      5859
I-PER       0.93      0.89      0.91      6528
B-ORG       0.86      0.86      0.86      3885
B-PER       0.95      0.93      0.94      4499

micro avg       0.89      0.89      0.89     36142
macro avg       0.89      0.89      0.89     36142
weighted avg       0.89      0.89      0.89     36142

processed 68754 tokens with 13369 phrases; found: 13165 phrases; correct: 11668.
accuracy:  89.13%; (non-O)
accuracy:  93.40%; precision:  88.63%; recall:  87.28%; FB1:  87.95
LOC: precision:  87.65%; recall:  86.26%; FB1:  86.95  4906
ORG: precision:  83.51%; recall:  82.88%; FB1:  83.19  3856
PER: precision:  94.21%; recall:  92.20%; FB1:  93.19  4403




###############################
lang:  he
precision    recall  f1-score   support

B-LOC       0.86      0.83      0.84      5160
I-ORG       0.79      0.82      0.80      6907
I-LOC       0.78      0.77      0.77      3133
I-PER       0.87      0.90      0.88      6816
B-ORG       0.79      0.74      0.76      4142
B-PER       0.85      0.87      0.86      4396

micro avg       0.83      0.83      0.83     30554
macro avg       0.82      0.82      0.82     30554
weighted avg       0.83      0.83      0.83     30554

processed 85418 tokens with 13698 phrases; found: 13352 phrases; correct: 10645.
accuracy:  83.01%; (non-O)
accuracy:  92.44%; precision:  79.73%; recall:  77.71%; FB1:  78.71
LOC: precision:  82.18%; recall:  80.00%; FB1:  81.08  5023
ORG: precision:  73.53%; recall:  68.32%; FB1:  70.83  3849
PER: precision:  82.30%; recall:  83.87%; FB1:  83.08  4480




###############################
lang:  hi
precision    recall  f1-score   support

B-LOC       0.84      0.86      0.85       414
I-ORG       0.91      0.88      0.90      1123
I-LOC       0.78      0.73      0.75       398
I-PER       0.85      0.92      0.88       598
B-ORG       0.89      0.86      0.87       364
B-PER       0.90      0.92      0.91       450

micro avg       0.87      0.87      0.87      3347
macro avg       0.86      0.86      0.86      3347
weighted avg       0.87      0.87      0.87      3347

processed 6005 tokens with 1228 phrases; found: 1239 phrases; correct: 1039.
accuracy:  87.00%; (non-O)
accuracy:  91.07%; precision:  83.86%; recall:  84.61%; FB1:  84.23
LOC: precision:  78.87%; recall:  81.16%; FB1:  80.00  426
ORG: precision:  84.94%; recall:  82.14%; FB1:  83.52  352
PER: precision:  87.64%; recall:  89.78%; FB1:  88.69  461




###############################
lang:  hu
precision    recall  f1-score   support

B-LOC       0.91      0.94      0.92      5671
I-ORG       0.89      0.91      0.90      5341
I-LOC       0.80      0.84      0.82      2404
I-PER       0.96      0.96      0.96      5501
B-ORG       0.90      0.86      0.88      3982
B-PER       0.96      0.95      0.95      4510

micro avg       0.91      0.92      0.92     27409
macro avg       0.90      0.91      0.91     27409
weighted avg       0.91      0.92      0.92     27409

processed 90302 tokens with 14163 phrases; found: 14084 phrases; correct: 12672.
accuracy:  91.85%; (non-O)
accuracy:  96.53%; precision:  89.97%; recall:  89.47%; FB1:  89.72
LOC: precision:  88.42%; recall:  90.78%; FB1:  89.58  5822
ORG: precision:  87.50%; recall:  83.12%; FB1:  85.25  3783
PER: precision:  94.08%; recall:  93.44%; FB1:  93.76  4479




###############################
lang:  id
precision    recall  f1-score   support

B-LOC       0.92      0.95      0.94      3745
I-ORG       0.91      0.93      0.92      8584
I-LOC       0.95      0.96      0.96      7809
I-PER       0.95      0.92      0.93      6520
B-ORG       0.91      0.89      0.90      3733
B-PER       0.94      0.93      0.93      3969

micro avg       0.93      0.93      0.93     34360
macro avg       0.93      0.93      0.93     34360
weighted avg       0.93      0.93      0.93     34360

processed 61834 tokens with 11447 phrases; found: 11423 phrases; correct: 10383.
accuracy:  93.31%; (non-O)
accuracy:  95.58%; precision:  90.90%; recall:  90.70%; FB1:  90.80
LOC: precision:  90.82%; recall:  93.56%; FB1:  92.17  3858
ORG: precision:  88.71%; recall:  86.93%; FB1:  87.81  3658
PER: precision:  93.01%; recall:  91.56%; FB1:  92.28  3907




###############################
lang:  it
precision    recall  f1-score   support

B-LOC       0.91      0.89      0.90      4820
I-ORG       0.89      0.91      0.90      9222
I-LOC       0.85      0.83      0.84      4366
I-PER       0.94      0.94      0.94      5794
B-ORG       0.89      0.87      0.88      4087
B-PER       0.96      0.96      0.96      4842

micro avg       0.91      0.90      0.90     33131
macro avg       0.90      0.90      0.90     33131
weighted avg       0.91      0.90      0.90     33131

processed 80871 tokens with 13749 phrases; found: 13514 phrases; correct: 12168.
accuracy:  90.32%; (non-O)
accuracy:  95.39%; precision:  90.04%; recall:  88.50%; FB1:  89.26
LOC: precision:  89.17%; recall:  86.62%; FB1:  87.88  4682
ORG: precision:  85.45%; recall:  83.31%; FB1:  84.37  3985
PER: precision:  94.66%; recall:  94.75%; FB1:  94.71  4847




###############################
lang:  ja
precision    recall  f1-score   support

B-LOC       0.74      0.77      0.76      5093
I-ORG       0.65      0.65      0.65     24814
I-LOC       0.72      0.77      0.75     17274
I-PER       0.77      0.79      0.78     21730
B-ORG       0.59      0.63      0.61      4267
B-PER       0.80      0.72      0.76      4081

micro avg       0.71      0.73      0.72     77259
macro avg       0.71      0.72      0.72     77259
weighted avg       0.71      0.73      0.72     77259

processed 306439 tokens with 13971 phrases; found: 13463 phrases; correct: 9267.
accuracy:  72.88%; (non-O)
accuracy:  88.72%; precision:  68.83%; recall:  66.33%; FB1:  67.56
LOC: precision:  71.74%; recall:  73.76%; FB1:  72.74  5298
ORG: precision:  59.48%; recall:  57.56%; FB1:  58.50  4514
PER: precision:  76.17%; recall:  66.96%; FB1:  71.27  3651




###############################
lang:  jv
precision    recall  f1-score   support

B-LOC       0.85      0.85      0.85        52
I-ORG       0.84      0.89      0.87        66
I-LOC       0.78      0.93      0.85        43
I-PER       0.93      0.95      0.94        44
B-ORG       0.79      0.78      0.78        40
B-PER       0.92      0.96      0.94        25

micro avg       0.85      0.89      0.87       270
macro avg       0.85      0.89      0.87       270
weighted avg       0.85      0.89      0.87       270

processed 678 tokens with 117 phrases; found: 117 phrases; correct: 95.
accuracy:  88.89%; (non-O)
accuracy:  92.92%; precision:  81.20%; recall:  81.20%; FB1:  81.20
LOC: precision:  78.85%; recall:  78.85%; FB1:  78.85  52
ORG: precision:  79.49%; recall:  77.50%; FB1:  78.48  39
PER: precision:  88.46%; recall:  92.00%; FB1:  90.20  26




###############################
lang:  ka
precision    recall  f1-score   support

B-LOC       0.86      0.90      0.88      5288
I-ORG       0.92      0.89      0.90      7800
I-LOC       0.76      0.84      0.80      2191
I-PER       0.91      0.95      0.93      4666
B-ORG       0.89      0.76      0.82      3807
B-PER       0.88      0.92      0.90      3962

micro avg       0.88      0.88      0.88     27714
macro avg       0.87      0.88      0.87     27714
weighted avg       0.88      0.88      0.88     27714

processed 81921 tokens with 13057 phrases; found: 12917 phrases; correct: 10903.
accuracy:  88.35%; (non-O)
accuracy:  94.72%; precision:  84.41%; recall:  83.50%; FB1:  83.95
LOC: precision:  83.37%; recall:  86.82%; FB1:  85.06  5507
ORG: precision:  84.01%; recall:  72.31%; FB1:  77.72  3277
PER: precision:  86.11%; recall:  89.83%; FB1:  87.93  4133




###############################
lang:  kk
precision    recall  f1-score   support

B-LOC       0.73      0.97      0.83       383
I-ORG       0.92      0.84      0.88       592
I-LOC       0.55      0.64      0.59       210
I-PER       0.90      0.97      0.93       466
B-ORG       0.86      0.64      0.74       355
B-PER       0.90      0.91      0.91       377

micro avg       0.83      0.85      0.84      2383
macro avg       0.81      0.83      0.81      2383
weighted avg       0.84      0.85      0.84      2383

processed 7936 tokens with 1115 phrases; found: 1157 phrases; correct: 858.
accuracy:  85.14%; (non-O)
accuracy:  93.47%; precision:  74.16%; recall:  76.95%; FB1:  75.53
LOC: precision:  61.06%; recall:  81.46%; FB1:  69.80  511
ORG: precision:  80.68%; recall:  60.00%; FB1:  68.82  264
PER: precision:  87.17%; recall:  88.33%; FB1:  87.75  382




###############################
lang:  ko
precision    recall  f1-score   support

B-LOC       0.88      0.91      0.89      5855
I-ORG       0.83      0.85      0.84      5437
I-LOC       0.83      0.88      0.85      2712
I-PER       0.87      0.88      0.88      3468
B-ORG       0.84      0.77      0.80      4319
B-PER       0.87      0.83      0.85      4249

micro avg       0.86      0.85      0.85     26040
macro avg       0.85      0.85      0.85     26040
weighted avg       0.86      0.85      0.85     26040

processed 80838 tokens with 14423 phrases; found: 14035 phrases; correct: 11713.
accuracy:  85.26%; (non-O)
accuracy:  93.54%; precision:  83.46%; recall:  81.21%; FB1:  82.32
LOC: precision:  85.66%; recall:  88.76%; FB1:  87.18  6067
ORG: precision:  78.26%; recall:  71.43%; FB1:  74.69  3942
PER: precision:  85.22%; recall:  80.75%; FB1:  82.92  4026




###############################
lang:  ml
precision    recall  f1-score   support

B-LOC       0.82      0.86      0.84       443
I-ORG       0.90      0.88      0.89       774
I-LOC       0.80      0.75      0.77       219
I-PER       0.89      0.93      0.91       492
B-ORG       0.86      0.77      0.81       354
B-PER       0.87      0.89      0.88       407

micro avg       0.87      0.86      0.87      2689
macro avg       0.86      0.85      0.85      2689
weighted avg       0.87      0.86      0.86      2689

processed 6727 tokens with 1204 phrases; found: 1195 phrases; correct: 974.
accuracy:  86.31%; (non-O)
accuracy:  92.81%; precision:  81.51%; recall:  80.90%; FB1:  81.20
LOC: precision:  79.00%; recall:  82.39%; FB1:  80.66  462
ORG: precision:  80.57%; recall:  71.47%; FB1:  75.75  314
PER: precision:  84.96%; recall:  87.47%; FB1:  86.20  419




###############################
lang:  mr
precision    recall  f1-score   support

B-LOC       0.85      0.86      0.86       525
I-ORG       0.89      0.93      0.91       852
I-LOC       0.71      0.67      0.69       258
I-PER       0.91      0.92      0.92       598
B-ORG       0.87      0.80      0.83       364
B-PER       0.86      0.92      0.89       375

micro avg       0.87      0.88      0.87      2972
macro avg       0.85      0.85      0.85      2972
weighted avg       0.87      0.88      0.87      2972

processed 7356 tokens with 1264 phrases; found: 1267 phrases; correct: 1066.
accuracy:  87.69%; (non-O)
accuracy:  93.50%; precision:  84.14%; recall:  84.34%; FB1:  84.24
LOC: precision:  83.27%; recall:  84.38%; FB1:  83.82  532
ORG: precision:  84.78%; recall:  78.02%; FB1:  81.26  335
PER: precision:  84.75%; recall:  90.40%; FB1:  87.48  400




###############################
lang:  ms
precision    recall  f1-score   support

B-LOC       0.94      0.98      0.96       367
I-ORG       0.90      0.91      0.91       913
I-LOC       0.96      0.98      0.97       898
I-PER       0.91      0.90      0.91       555
B-ORG       0.90      0.86      0.88       375
B-PER       0.91      0.92      0.92       373

micro avg       0.92      0.93      0.93      3481
macro avg       0.92      0.93      0.92      3481
weighted avg       0.92      0.93      0.93      3481

processed 5874 tokens with 1115 phrases; found: 1120 phrases; correct: 1010.
accuracy:  93.08%; (non-O)
accuracy:  94.91%; precision:  90.18%; recall:  90.58%; FB1:  90.38
LOC: precision:  93.72%; recall:  97.55%; FB1:  95.59  382
ORG: precision:  87.22%; recall:  83.73%; FB1:  85.44  360
PER: precision:  89.42%; recall:  90.62%; FB1:  90.01  378




###############################
lang:  my
precision    recall  f1-score   support

B-LOC       0.61      0.93      0.74        56
I-ORG       0.87      0.71      0.78        68
I-LOC       0.15      1.00      0.26         4
I-PER       0.85      0.63      0.72        46
B-ORG       0.90      0.58      0.70        33
B-PER       0.90      0.60      0.72        30

micro avg       0.70      0.72      0.71       237
macro avg       0.72      0.74      0.65       237
weighted avg       0.80      0.72      0.73       237

processed 756 tokens with 119 phrases; found: 126 phrases; correct: 83.
accuracy:  71.73%; (non-O)
accuracy:  86.77%; precision:  65.87%; recall:  69.75%; FB1:  67.76
LOC: precision:  60.00%; recall:  91.07%; FB1:  72.34  85
ORG: precision:  80.95%; recall:  51.52%; FB1:  62.96  21
PER: precision:  75.00%; recall:  50.00%; FB1:  60.00  20




###############################
lang:  nl
precision    recall  f1-score   support

B-LOC       0.89      0.93      0.91      5133
I-ORG       0.90      0.88      0.89      6693
I-LOC       0.86      0.86      0.86      3662
I-PER       0.95      0.94      0.95      6371
B-ORG       0.89      0.85      0.87      3908
B-PER       0.96      0.94      0.95      4684

micro avg       0.91      0.90      0.91     30451
macro avg       0.91      0.90      0.90     30451
weighted avg       0.91      0.90      0.91     30451

processed 85122 tokens with 13725 phrases; found: 13653 phrases; correct: 12219.
accuracy:  90.42%; (non-O)
accuracy:  96.01%; precision:  89.50%; recall:  89.03%; FB1:  89.26
LOC: precision:  87.24%; recall:  90.71%; FB1:  88.94  5337
ORG: precision:  86.72%; recall:  82.19%; FB1:  84.39  3704
PER: precision:  94.34%; recall:  92.89%; FB1:  93.61  4612




###############################
lang:  pt
precision    recall  f1-score   support

B-LOC       0.91      0.92      0.92      4779
I-ORG       0.89      0.92      0.91     10542
I-LOC       0.88      0.89      0.88      6467
I-PER       0.96      0.92      0.94      7310
B-ORG       0.89      0.88      0.88      3753
B-PER       0.95      0.93      0.94      4291

micro avg       0.91      0.91      0.91     37142
macro avg       0.91      0.91      0.91     37142
weighted avg       0.91      0.91      0.91     37142

processed 63647 tokens with 12823 phrases; found: 12725 phrases; correct: 11471.
accuracy:  91.00%; (non-O)
accuracy:  94.15%; precision:  90.15%; recall:  89.46%; FB1:  89.80
LOC: precision:  89.40%; recall:  90.86%; FB1:  90.12  4857
ORG: precision:  86.02%; recall:  84.95%; FB1:  85.48  3706
PER: precision:  94.69%; recall:  91.84%; FB1:  93.25  4162




###############################
lang:  ru
precision    recall  f1-score   support

B-LOC       0.88      0.90      0.89      4560
I-ORG       0.89      0.86      0.88      8008
I-LOC       0.83      0.86      0.84      3060
I-PER       0.95      0.97      0.96      7544
B-ORG       0.88      0.80      0.84      4074
B-PER       0.92      0.96      0.94      3543

micro avg       0.90      0.90      0.90     30789
macro avg       0.89      0.89      0.89     30789
weighted avg       0.90      0.90      0.90     30789

processed 71288 tokens with 12177 phrases; found: 12036 phrases; correct: 10465.
accuracy:  89.74%; (non-O)
accuracy:  94.44%; precision:  86.95%; recall:  85.94%; FB1:  86.44
LOC: precision:  86.28%; recall:  88.53%; FB1:  87.39  4679
ORG: precision:  83.80%; recall:  75.41%; FB1:  79.38  3666
PER: precision:  90.92%; recall:  94.72%; FB1:  92.78  3691




###############################
lang:  sw
precision    recall  f1-score   support

B-LOC       0.83      0.94      0.88       388
I-ORG       0.86      0.79      0.82       763
I-LOC       0.76      0.88      0.81       568
I-PER       0.95      0.95      0.95       744
B-ORG       0.91      0.82      0.86       374
B-PER       0.95      0.95      0.95       432

micro avg       0.87      0.88      0.88      3269
macro avg       0.88      0.89      0.88      3269
weighted avg       0.88      0.88      0.88      3269

processed 5786 tokens with 1194 phrases; found: 1209 phrases; correct: 1042.
accuracy:  88.35%; (non-O)
accuracy:  92.02%; precision:  86.19%; recall:  87.27%; FB1:  86.72
LOC: precision:  77.63%; recall:  87.63%; FB1:  82.32  438
ORG: precision:  88.17%; recall:  79.68%; FB1:  83.71  338
PER: precision:  93.30%; recall:  93.52%; FB1:  93.41  433




###############################
lang:  ta
precision    recall  f1-score   support

B-LOC       0.82      0.86      0.84       436
I-ORG       0.84      0.87      0.85       814
I-LOC       0.76      0.68      0.72       239
I-PER       0.90      0.94      0.92       615
B-ORG       0.82      0.77      0.79       383
B-PER       0.86      0.90      0.88       422

micro avg       0.84      0.86      0.85      2909
macro avg       0.83      0.84      0.83      2909
weighted avg       0.84      0.86      0.85      2909

processed 7234 tokens with 1241 phrases; found: 1252 phrases; correct: 1006.
accuracy:  85.87%; (non-O)
accuracy:  92.31%; precision:  80.35%; recall:  81.06%; FB1:  80.71
LOC: precision:  80.04%; recall:  83.72%; FB1:  81.84  456
ORG: precision:  76.26%; recall:  71.28%; FB1:  73.68  358
PER: precision:  84.02%; recall:  87.20%; FB1:  85.58  438




###############################
lang:  te
precision    recall  f1-score   support

B-LOC       0.77      0.89      0.82       450
I-ORG       0.87      0.79      0.83       633
I-LOC       0.67      0.82      0.74       178
I-PER       0.80      0.86      0.83       294
B-ORG       0.75      0.70      0.72       340
B-PER       0.79      0.80      0.80       381

micro avg       0.79      0.81      0.80      2276
macro avg       0.78      0.81      0.79      2276
weighted avg       0.80      0.81      0.80      2276

processed 8155 tokens with 1171 phrases; found: 1226 phrases; correct: 890.
accuracy:  81.06%; (non-O)
accuracy:  92.36%; precision:  72.59%; recall:  76.00%; FB1:  74.26
LOC: precision:  73.04%; recall:  84.89%; FB1:  78.52  523
ORG: precision:  69.09%; recall:  64.41%; FB1:  66.67  317
PER: precision:  74.87%; recall:  75.85%; FB1:  75.36  386




###############################
lang:  th
precision    recall  f1-score   support

B-LOC       0.75      0.72      0.74      6430
I-ORG       0.67      0.71      0.69     56669
I-LOC       0.75      0.80      0.77     47216
I-PER       0.76      0.78      0.77     57226
B-ORG       0.55      0.53      0.54      5136
B-PER       0.45      0.62      0.53      5297

micro avg       0.71      0.75      0.73    177974
macro avg       0.66      0.69      0.67    177974
weighted avg       0.71      0.75      0.73    177974

processed 626147 tokens with 20775 phrases; found: 18317 phrases; correct: 12096.
accuracy:  74.70%; (non-O)
accuracy:  88.27%; precision:  66.04%; recall:  58.22%; FB1:  61.88
LOC: precision:  67.74%; recall:  63.63%; FB1:  65.62  6144
ORG: precision:  55.11%; recall:  45.64%; FB1:  49.93  4916
PER: precision:  72.00%; recall:  62.96%; FB1:  67.18  7257




###############################
lang:  tl
precision    recall  f1-score   support

B-LOC       0.87      0.90      0.88       327
I-ORG       0.88      0.93      0.90      1045
I-LOC       0.93      0.85      0.89       706
I-PER       0.91      0.89      0.90       813
B-ORG       0.87      0.90      0.89       341
B-PER       0.90      0.90      0.90       366

micro avg       0.90      0.90      0.90      3598
macro avg       0.89      0.90      0.90      3598
weighted avg       0.90      0.90      0.90      3598

processed 4627 tokens with 1034 phrases; found: 1057 phrases; correct: 908.
accuracy:  89.88%; (non-O)
accuracy:  91.16%; precision:  85.90%; recall:  87.81%; FB1:  86.85
LOC: precision:  81.82%; recall:  85.32%; FB1:  83.53  341
ORG: precision:  86.04%; recall:  88.56%; FB1:  87.28  351
PER: precision:  89.59%; recall:  89.34%; FB1:  89.47  365




###############################
lang:  tr
precision    recall  f1-score   support

B-LOC       0.91      0.91      0.91      4914
I-ORG       0.88      0.94      0.91      6979
I-LOC       0.85      0.83      0.84      3005
I-PER       0.95      0.94      0.94      5694
B-ORG       0.89      0.89      0.89      4154
B-PER       0.95      0.93      0.94      4519

micro avg       0.91      0.92      0.91     29265
macro avg       0.91      0.91      0.91     29265
weighted avg       0.91      0.92      0.91     29265

processed 75731 tokens with 13587 phrases; found: 13482 phrases; correct: 12147.
accuracy:  91.66%; (non-O)
accuracy:  95.91%; precision:  90.10%; recall:  89.40%; FB1:  89.75
LOC: precision:  88.93%; recall:  88.93%; FB1:  88.93  4914
ORG: precision:  87.10%; recall:  87.10%; FB1:  87.10  4154
PER: precision:  94.22%; recall:  92.03%; FB1:  93.12  4414




###############################
lang:  ur
precision    recall  f1-score   support

B-LOC       0.87      0.94      0.90       334
I-ORG       0.95      0.85      0.90      1005
I-LOC       0.86      0.96      0.91       904
I-PER       0.93      0.97      0.95       928
B-ORG       0.96      0.84      0.89       323
B-PER       0.93      0.95      0.94       363

micro avg       0.91      0.92      0.92      3857
macro avg       0.92      0.92      0.92      3857
weighted avg       0.92      0.92      0.92      3857

processed 5027 tokens with 1020 phrases; found: 1017 phrases; correct: 916.
accuracy:  92.20%; (non-O)
accuracy:  93.06%; precision:  90.07%; recall:  89.80%; FB1:  89.94
LOC: precision:  85.08%; recall:  92.22%; FB1:  88.51  362
ORG: precision:  95.74%; recall:  83.59%; FB1:  89.26  282
PER: precision:  90.62%; recall:  93.11%; FB1:  91.85  373




###############################
lang:  vi
precision    recall  f1-score   support

B-LOC       0.89      0.92      0.91      3717
I-ORG       0.90      0.92      0.91     13562
I-LOC       0.90      0.91      0.90      8018
I-PER       0.92      0.91      0.92      7787
B-ORG       0.90      0.86      0.88      3704
B-PER       0.92      0.93      0.93      3884

micro avg       0.91      0.91      0.91     40672
macro avg       0.91      0.91      0.91     40672
weighted avg       0.91      0.91      0.91     40672

processed 64967 tokens with 11305 phrases; found: 11317 phrases; correct: 9984.
accuracy:  91.01%; (non-O)
accuracy:  93.30%; precision:  88.22%; recall:  88.31%; FB1:  88.27
LOC: precision:  86.64%; recall:  90.23%; FB1:  88.40  3871
ORG: precision:  86.90%; recall:  83.26%; FB1:  85.04  3549
PER: precision:  90.99%; recall:  91.30%; FB1:  91.15  3897




###############################
lang:  yo
precision    recall  f1-score   support

B-LOC       0.55      0.72      0.62        39
I-ORG       0.53      0.23      0.32        87
I-LOC       0.68      0.83      0.75        72
I-PER       0.82      0.66      0.73        71
B-ORG       0.50      0.28      0.36        29
B-PER       0.85      0.79      0.82        43

micro avg       0.68      0.58      0.62       341
macro avg       0.66      0.58      0.60       341
weighted avg       0.66      0.58      0.60       341

processed 503 tokens with 111 phrases; found: 107 phrases; correct: 66.
accuracy:  57.77%; (non-O)
accuracy:  71.17%; precision:  61.68%; recall:  59.46%; FB1:  60.55
LOC: precision:  52.94%; recall:  69.23%; FB1:  60.00  51
ORG: precision:  50.00%; recall:  27.59%; FB1:  35.56  16
PER: precision:  77.50%; recall:  72.09%; FB1:  74.70  40




###############################
lang:  zh
precision    recall  f1-score   support

B-LOC       0.76      0.84      0.80      4371
I-ORG       0.77      0.76      0.77     17399
I-LOC       0.78      0.86      0.82     12282
I-PER       0.87      0.87      0.87     12897
B-ORG       0.70      0.71      0.70      3779
B-PER       0.86      0.82      0.84      3899

micro avg       0.80      0.82      0.81     54627
macro avg       0.79      0.81      0.80     54627
weighted avg       0.80      0.82      0.81     54627

processed 207418 tokens with 12532 phrases; found: 12410 phrases; correct: 9536.
accuracy:  81.65%; (non-O)
accuracy:  91.91%; precision:  76.84%; recall:  76.09%; FB1:  76.47
LOC: precision:  74.87%; recall:  80.78%; FB1:  77.71  4827
ORG: precision:  71.48%; recall:  66.88%; FB1:  69.10  3850
PER: precision:  84.92%; recall:  80.40%; FB1:  82.60  3733



```
