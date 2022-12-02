---
layout: model
title: Detect Entities in 8 languages - WIKINER (ner_wikiner_glove_840B_300)
author: John Snow Labs
name: ner_wikiner_glove_840B_300
date: 2021-07-19
tags: [open_source, xx, multilingual, glove, ner]
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
We used WordEmbeddings (glove_840B_300) model for the embeddings to train this NER model.

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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_wikiner_glove_840B_300_xx_3.1.3_2.4_1626717450663.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

ner_model = NerDLModel.pretrained('ner_wikiner_glove_840B_300', 'xx') \
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

val ner_model = NerDLModel.pretrained("ner_wikiner_glove_840B_300", "xx") 
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

ner_df = nlu.load('xx.ner.ner_wikiner_glove_840B_300').predict(text, output_level='token')
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
|Hohen Karlsschule|LOC      |
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
|Model Name:|ner_wikiner_glove_840B_300|
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

processed 1267027 tokens with 134558 phrases; found: 132064 phrases; correct: 112519.
accuracy:  84.14%; (non-O)
accuracy:  96.92%; precision:  85.20%; recall:  83.62%; FB1:  84.40
LOC: precision:  84.89%; recall:  88.12%; FB1:  86.48  59436
MISC: precision:  78.88%; recall:  67.35%; FB1:  72.66  19886
ORG: precision:  81.37%; recall:  70.97%; FB1:  75.82  13014
PER: precision:  90.08%; recall:  91.56%; FB1:  90.81  39728

Language by language benchmarks (multi-label classification and CoNLL Eval):

lang:  english
precision    recall  f1-score   support

B-LOC       0.85      0.90      0.87      8600
I-ORG       0.78      0.79      0.78      4249
I-LOC       0.83      0.79      0.81      3960
I-PER       0.94      0.93      0.94      4472
B-ORG       0.84      0.76      0.80      4882
B-PER       0.93      0.94      0.93      9639

micro avg       0.87      0.87      0.87     35802
macro avg       0.86      0.85      0.85     35802
weighted avg       0.87      0.87      0.87     35802

processed 349486 tokens with 30471 phrases; found: 29911 phrases; correct: 25093.
accuracy:  84.41%; (non-O)
accuracy:  97.30%; precision:  83.89%; recall:  82.35%; FB1:  83.11
LOC: precision:  83.01%; recall:  87.38%; FB1:  85.14  9053
MISC: precision:  75.17%; recall:  68.38%; FB1:  71.62  6686
ORG: precision:  80.78%; recall:  72.82%; FB1:  76.59  4401
PER: precision:  92.08%; recall:  93.34%; FB1:  92.70  9771




###############################
lang:  french
precision    recall  f1-score   support

B-LOC       0.81      0.87      0.84     11482
I-ORG       0.81      0.74      0.77      2143
I-LOC       0.75      0.60      0.67      4495
I-PER       0.95      0.94      0.95      5339
B-ORG       0.86      0.78      0.82      2556
B-PER       0.92      0.94      0.93      7524

micro avg       0.86      0.85      0.85     33539
macro avg       0.85      0.81      0.83     33539
weighted avg       0.86      0.85      0.85     33539

processed 348522 tokens with 25499 phrases; found: 25298 phrases; correct: 21261.
accuracy:  80.86%; (non-O)
accuracy:  97.44%; precision:  84.04%; recall:  83.38%; FB1:  83.71
LOC: precision:  80.06%; recall:  85.79%; FB1:  82.83  12303
MISC: precision:  82.42%; recall:  63.25%; FB1:  71.57  3021
ORG: precision:  83.49%; recall:  75.59%; FB1:  79.34  2314
PER: precision:  91.24%; recall:  92.89%; FB1:  92.06  7660




###############################
lang:  german
precision    recall  f1-score   support

B-LOC       0.85      0.87      0.86     20709
I-ORG       0.82      0.82      0.82      5933
I-LOC       0.77      0.79      0.78      6405
I-PER       0.95      0.97      0.96      8365
B-ORG       0.83      0.73      0.78      6759
B-PER       0.92      0.93      0.92     10647

micro avg       0.86      0.87      0.86     58818
macro avg       0.86      0.85      0.85     58818
weighted avg       0.86      0.87      0.86     58818

processed 349393 tokens with 46006 phrases; found: 44918 phrases; correct: 37486.
accuracy:  83.53%; (non-O)
accuracy:  95.57%; precision:  83.45%; recall:  81.48%; FB1:  82.46
LOC: precision:  82.80%; recall:  85.49%; FB1:  84.13  21382
MISC: precision:  77.46%; recall:  66.53%; FB1:  71.58  6778
ORG: precision:  79.88%; recall:  70.72%; FB1:  75.02  5984
PER: precision:  90.50%; recall:  91.58%; FB1:  91.04  10774




###############################
lang:  italian
precision    recall  f1-score   support

B-LOC       0.88      0.92      0.90     13050
I-ORG       0.78      0.71      0.74      1211
I-LOC       0.89      0.85      0.87      7454
I-PER       0.93      0.94      0.94      4539
B-ORG       0.88      0.72      0.79      2222
B-PER       0.90      0.93      0.92      7206

micro avg       0.89      0.89      0.89     35682
macro avg       0.88      0.85      0.86     35682
weighted avg       0.89      0.89      0.89     35682

processed 349242 tokens with 26227 phrases; found: 25988 phrases; correct: 22529.
accuracy:  85.99%; (non-O)
accuracy:  98.06%; precision:  86.69%; recall:  85.90%; FB1:  86.29
LOC: precision:  86.33%; recall:  90.53%; FB1:  88.38  13685
MISC: precision:  81.88%; recall:  67.03%; FB1:  73.72  3069
ORG: precision:  85.91%; recall:  70.52%; FB1:  77.46  1824
PER: precision:  89.54%; recall:  92.08%; FB1:  90.79  7410




###############################
lang:  polish
precision    recall  f1-score   support

B-LOC       0.86      0.91      0.88     17757
I-ORG       0.80      0.69      0.74      2105
I-LOC       0.83      0.78      0.80      5242
I-PER       0.88      0.94      0.91      6672
B-ORG       0.87      0.71      0.78      3700
B-PER       0.88      0.89      0.88      9670

micro avg       0.86      0.87      0.87     45146
macro avg       0.85      0.82      0.83     45146
weighted avg       0.86      0.87      0.86     45146

processed 350132 tokens with 36235 phrases; found: 35498 phrases; correct: 30071.
accuracy:  83.07%; (non-O)
accuracy:  97.12%; precision:  84.71%; recall:  82.99%; FB1:  83.84
LOC: precision:  84.80%; recall:  89.59%; FB1:  87.13  18761
MISC: precision:  78.18%; recall:  60.32%; FB1:  68.10  3941
ORG: precision:  86.15%; recall:  70.27%; FB1:  77.40  3018
PER: precision:  86.74%; recall:  87.70%; FB1:  87.22  9778




###############################
lang:  portuguese
precision    recall  f1-score   support

B-LOC       0.91      0.94      0.92     14818
I-ORG       0.84      0.74      0.79      1705
I-LOC       0.89      0.88      0.89      8354
I-PER       0.94      0.93      0.93      4338
B-ORG       0.90      0.77      0.83      2351
B-PER       0.92      0.93      0.93      6398

micro avg       0.90      0.90      0.90     37964
macro avg       0.90      0.87      0.88     37964
weighted avg       0.90      0.90      0.90     37964

processed 348966 tokens with 26513 phrases; found: 26359 phrases; correct: 23574.
accuracy:  88.48%; (non-O)
accuracy:  98.39%; precision:  89.43%; recall:  88.91%; FB1:  89.17
LOC: precision:  89.52%; recall:  92.60%; FB1:  91.04  15328
MISC: precision:  84.55%; recall:  72.47%; FB1:  78.05  2525
ORG: precision:  88.53%; recall:  75.84%; FB1:  81.70  2014
PER: precision:  91.40%; recall:  92.75%; FB1:  92.07  6492




###############################
lang:  russian
precision    recall  f1-score   support

B-LOC       0.91      0.93      0.92     14707
I-ORG       0.78      0.64      0.70      2594
I-LOC       0.86      0.79      0.82      5047
I-PER       0.96      0.95      0.95      6366
B-ORG       0.87      0.75      0.81      3697
B-PER       0.88      0.91      0.90      7119

micro avg       0.89      0.87      0.88     39530
macro avg       0.88      0.83      0.85     39530
weighted avg       0.89      0.87      0.88     39530



###############################
lang:  spanish
precision    recall  f1-score   support

B-LOC       0.86      0.90      0.88     11963
I-ORG       0.82      0.78      0.80      1950
I-LOC       0.84      0.81      0.83      6162
I-PER       0.95      0.93      0.94      4678
B-ORG       0.83      0.77      0.80      2084
B-PER       0.93      0.94      0.94      7215

micro avg       0.88      0.88      0.88     34052
macro avg       0.87      0.86      0.86     34052
weighted avg       0.88      0.88      0.88     34052

processed 348209 tokens with 24505 phrases; found: 24375 phrases; correct: 21187.
accuracy:  85.54%; (non-O)
accuracy:  98.07%; precision:  86.92%; recall:  86.46%; FB1:  86.69
LOC: precision:  85.32%; recall:  89.28%; FB1:  87.25  12518
MISC: precision:  82.54%; recall:  67.78%; FB1:  74.43  2663
ORG: precision:  81.85%; recall:  76.39%; FB1:  79.03  1945
PER: precision:  92.66%; recall:  93.10%; FB1:  92.88  7249

```
