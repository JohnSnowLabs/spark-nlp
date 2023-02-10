---
layout: model
title: Detect Entities in 8 languages - WIKINER (ner_wikiner_xlm_roberta_base)
author: John Snow Labs
name: ner_wikiner_xlm_roberta_base
date: 2021-07-19
tags: [open_source, ner, multilingual, xlm_roberta, xx, wikiner]
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

This NER model was trained over WIKINER datasets with 8 languages including `English`, `French`, `German`, `Italian`, `Polish`, `Portuguese`, `Russian`, and `Spanish`.
We used XlmRoBertaEmbeddings (xlm_roberta_base) model for the embeddings to train this NER model.

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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_wikiner_xlm_roberta_base_xx_3.1.3_2.4_1626719349008.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ner_wikiner_xlm_roberta_base_xx_3.1.3_2.4_1626719349008.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
    .pretrained('xlm_roberta_base', 'xx')\
    .setInputCols(["token", "document"])\
    .setOutputCol("embeddings")

ner_model = NerDLModel.pretrained('ner_wikiner_xlm_roberta_base', 'xx') \
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

val embeddings = XlmRoBertaEmbeddings.pretrained("xlm_roberta_base", "xx")
    .setInputCols(Array("document", "token")) 
    .setOutputCol("embeddings")

val ner_model = NerDLModel.pretrained("ner_wikiner_xlm_roberta_base", "xx") 
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

ner_df = nlu.load('xx.ner.ner_wikiner_xlm_roberta_base').predict(text, output_level='token')
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
|Starbucks        |LOC      |
+-----------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_wikiner_xlm_roberta_base|
|Type:|ner|
|Compatibility:|Spark NLP 3.1.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|xx|

## Data Source

[https://figshare.com/articles/dataset/Learning_multilingual_named_entity_recognition_from_Wikipedia/5462500](https://figshare.com/articles/dataset/Learning_multilingual_named_entity_recognition_from_Wikipedia/5462500)

## Benchmarking

```bash
Average of all languages benchmark (multi-label classification and CoNLL Eval):

processed 1267026 tokens with 134558 phrases; found: 132447 phrases; correct: 114590.
accuracy:  85.26%; (non-O)
accuracy:  97.23%; precision:  86.52%; recall:  85.16%; FB1:  85.83
LOC: precision:  87.26%; recall:  88.62%; FB1:  87.94  58155
MISC: precision:  80.06%; recall:  70.23%; FB1:  74.82  20432
ORG: precision:  80.02%; recall:  75.09%; FB1:  77.48  14003
PER: precision:  91.03%; recall:  92.83%; FB1:  91.92  39857



Language by language benchmarks (multi-label classification and CoNLL Eval):


lang:  english
precision    recall  f1-score   support

B-LOC       0.84      0.91      0.87      8600
I-ORG       0.82      0.81      0.82      4249
I-LOC       0.84      0.82      0.83      3960
I-PER       0.95      0.94      0.95      4472
B-ORG       0.83      0.77      0.80      4882
B-PER       0.93      0.94      0.94      9639

micro avg       0.88      0.88      0.88     35802
macro avg       0.87      0.87      0.87     35802
weighted avg       0.88      0.88      0.88     35802

processed 349485 tokens with 30471 phrases; found: 30143 phrases; correct: 25648.
accuracy:  85.02%; (non-O)
accuracy:  97.68%; precision:  85.09%; recall:  84.17%; FB1:  84.63
LOC: precision:  82.69%; recall:  88.64%; FB1:  85.56  9219
MISC: precision:  80.26%; recall:  71.82%; FB1:  75.81  6577
ORG: precision:  81.13%; recall:  75.91%; FB1:  78.43  4568
PER: precision:  92.44%; recall:  93.79%; FB1:  93.11  9779




###############################
lang:  french
precision    recall  f1-score   support

B-LOC       0.84      0.86      0.85     11482
I-ORG       0.80      0.77      0.78      2143
I-LOC       0.81      0.60      0.69      4495
I-PER       0.97      0.94      0.95      5339
B-ORG       0.84      0.81      0.82      2556
B-PER       0.93      0.93      0.93      7524

micro avg       0.87      0.85      0.86     33539
macro avg       0.86      0.82      0.84     33539
weighted avg       0.87      0.85      0.86     33539

processed 348522 tokens with 25499 phrases; found: 25270 phrases; correct: 21525.
accuracy:  82.17%; (non-O)
accuracy:  97.62%; precision:  85.18%; recall:  84.42%; FB1:  84.80
LOC: precision:  82.63%; recall:  85.18%; FB1:  83.88  11836
MISC: precision:  80.10%; recall:  69.62%; FB1:  74.49  3422
ORG: precision:  82.91%; recall:  79.34%; FB1:  81.09  2446
PER: precision:  92.20%; recall:  92.72%; FB1:  92.46  7566




###############################
lang:  german
precision    recall  f1-score   support

B-LOC       0.88      0.90      0.89     20709
I-ORG       0.82      0.87      0.85      5933
I-LOC       0.81      0.82      0.82      6405
I-PER       0.96      0.97      0.97      8365
B-ORG       0.82      0.80      0.81      6759
B-PER       0.93      0.94      0.94     10647

micro avg       0.88      0.89      0.89     58818
macro avg       0.87      0.88      0.88     58818
weighted avg       0.88      0.89      0.89     58818

processed 349393 tokens with 46006 phrases; found: 45517 phrases; correct: 39247.
accuracy:  86.81%; (non-O)
accuracy:  96.44%; precision:  86.22%; recall:  85.31%; FB1:  85.76
LOC: precision:  86.58%; recall:  88.33%; FB1:  87.45  21128
MISC: precision:  81.40%; recall:  72.67%; FB1:  76.79  7044
ORG: precision:  80.11%; recall:  77.76%; FB1:  78.92  6561
PER: precision:  92.40%; recall:  93.59%; FB1:  92.99  10784




###############################
lang:  italian
precision    recall  f1-score   support

B-LOC       0.91      0.92      0.91     13050
I-ORG       0.75      0.81      0.78      1211
I-LOC       0.92      0.86      0.89      7454
I-PER       0.96      0.95      0.95      4539
B-ORG       0.84      0.84      0.84      2222
B-PER       0.93      0.94      0.93      7206

micro avg       0.91      0.90      0.91     35682
macro avg       0.88      0.88      0.88     35682
weighted avg       0.91      0.90      0.91     35682

processed 349242 tokens with 26227 phrases; found: 25982 phrases; correct: 23079.
accuracy:  88.06%; (non-O)
accuracy:  98.35%; precision:  88.83%; recall:  88.00%; FB1:  88.41
LOC: precision:  89.87%; recall:  90.22%; FB1:  90.05  13101
MISC: precision:  81.60%; recall:  74.05%; FB1:  77.64  3402
ORG: precision:  83.26%; recall:  82.36%; FB1:  82.81  2198
PER: precision:  92.01%; recall:  92.96%; FB1:  92.48  7281




###############################
lang:  polish
precision    recall  f1-score   support

B-LOC       0.91      0.93      0.92     17757
I-ORG       0.79      0.85      0.82      2105
I-LOC       0.89      0.85      0.87      5242
I-PER       0.97      0.95      0.96      6672
B-ORG       0.86      0.84      0.85      3700
B-PER       0.93      0.94      0.94      9670

micro avg       0.91      0.91      0.91     45146
macro avg       0.89      0.89      0.89     45146
weighted avg       0.91      0.91      0.91     45146

processed 350132 tokens with 36235 phrases; found: 35886 phrases; correct: 32107.
accuracy:  88.31%; (non-O)
accuracy:  97.98%; precision:  89.47%; recall:  88.61%; FB1:  89.04
LOC: precision:  90.17%; recall:  92.52%; FB1:  91.33  18221
MISC: precision:  82.48%; recall:  70.71%; FB1:  76.15  4379
ORG: precision:  85.37%; recall:  82.51%; FB1:  83.92  3576
PER: precision:  92.82%; recall:  93.21%; FB1:  93.01  9710




###############################
lang:  portuguese
precision    recall  f1-score   support

B-LOC       0.93      0.93      0.93     14818
I-ORG       0.80      0.85      0.83      1705
I-LOC       0.92      0.88      0.90      8354
I-PER       0.96      0.94      0.95      4338
B-ORG       0.84      0.86      0.85      2351
B-PER       0.94      0.94      0.94      6398

micro avg       0.92      0.91      0.92     37964
macro avg       0.90      0.90      0.90     37964
weighted avg       0.92      0.91      0.92     37964

processed 348966 tokens with 26513 phrases; found: 26349 phrases; correct: 23958.
accuracy:  90.10%; (non-O)
accuracy:  98.60%; precision:  90.93%; recall:  90.36%; FB1:  90.64
LOC: precision:  92.13%; recall:  91.97%; FB1:  92.05  14792
MISC: precision:  84.09%; recall:  79.46%; FB1:  81.71  2784
ORG: precision:  83.51%; recall:  85.28%; FB1:  84.39  2401
PER: precision:  93.91%; recall:  93.53%; FB1:  93.72  6372




###############################
lang:  russian
precision    recall  f1-score   support

B-LOC       0.93      0.95      0.94     14707
I-ORG       0.84      0.73      0.78      2594
I-LOC       0.86      0.87      0.87      5047
I-PER       0.98      0.96      0.97      6366
B-ORG       0.86      0.84      0.85      3697
B-PER       0.94      0.95      0.94      7119

micro avg       0.92      0.92      0.92     39530
macro avg       0.90      0.88      0.89     39530
weighted avg       0.92      0.92      0.92     39530




###############################
lang:  spanish
precision    recall  f1-score   support

B-LOC       0.89      0.90      0.89     11963
I-ORG       0.83      0.79      0.81      1950
I-LOC       0.89      0.80      0.84      6162
I-PER       0.97      0.94      0.95      4678
B-ORG       0.84      0.80      0.82      2084
B-PER       0.94      0.94      0.94      7215

micro avg       0.90      0.88      0.89     34052
macro avg       0.89      0.86      0.88     34052
weighted avg       0.90      0.88      0.89     34052

processed 348209 tokens with 24505 phrases; found: 24360 phrases; correct: 21446.
accuracy:  86.63%; (non-O)
accuracy:  98.24%; precision:  88.04%; recall:  87.52%; FB1:  87.78
LOC: precision:  87.90%; recall:  88.74%; FB1:  88.32  12078
MISC: precision:  79.02%; recall:  74.68%; FB1:  76.79  3065
ORG: precision:  82.57%; recall:  78.89%; FB1:  80.69  1991
PER: precision:  93.61%; recall:  93.75%; FB1:  93.68  7226
```
