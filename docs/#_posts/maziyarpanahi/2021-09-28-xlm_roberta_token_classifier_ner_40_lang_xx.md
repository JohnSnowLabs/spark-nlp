---
layout: model
title: XLM-RoBERTa NER (Base, 40 languages)
author: John Snow Labs
name: xlm_roberta_token_classifier_ner_40_lang
date: 2021-09-28
tags: [xlm_roberta, multilingual, xtreme, ner, token_classification, xx, open_source]
task: Named Entity Recognition
language: xx
edition: Spark NLP 3.3.0
spark_version: 3.0
supported: true
recommended: true
annotator: XlmRoBertaForTokenClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

# XLM-R + NER

This model is a fine-tuned  [XLM-Roberta-base](https://arxiv.org/abs/1911.02116) over the 40 languages proposed in [XTREME](https://github.com/google-research/xtreme) from [Wikiann](https://aclweb.org/anthology/P17-1178). 

The covered labels are:

```
LOC
ORG
PER
O
```

## Predicted Entities

`LOC`, `ORG`, `PER`, `O`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_token_classifier_ner_40_lang_xx_3.3.0_3.0_1632835906778.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = XlmRoBertaForTokenClassification \
.pretrained('xlm_roberta_token_classifier_ner_40_lang', 'xx') \
.setInputCols(['token', 'document']) \
.setOutputCol('ner') \
.setCaseSensitive(True) \
.setMaxSentenceLength(512)

# since output column is IOB/IOB2 style, NerConverter can extract entities
ner_converter = NerConverter() \
.setInputCols(['document', 'token', 'ner']) \
.setOutputCol('entities')

pipeline = Pipeline(stages=[
document_assembler, 
tokenizer,
tokenClassifier,
ner_converter
])

example = spark.createDataFrame([['My name is John!']]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = Tokenizer() 
.setInputCols("document") 
.setOutputCol("token")

val tokenClassifier = XlmRoBertaForTokenClassification.pretrained("xlm_roberta_token_classifier_ner_40_lang", "xx")
.setInputCols("document", "token")
.setOutputCol("ner")
.setCaseSensitive(true)
.setMaxSentenceLength(512)

// since output column is IOB/IOB2 style, NerConverter can extract entities
val ner_converter = NerConverter() 
.setInputCols("document", "token", "ner") 
.setOutputCol("entities")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, tokenClassifier, ner_converter))

val example = Seq.empty["My name is John!"].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("xx.classify.token_xlm_roberta.token_classifier_ner_40_lang").predict("""My name is John!""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_token_classifier_ner_40_lang|
|Compatibility:|Spark NLP 3.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[ner]|
|Language:|xx|
|Case sensitive:|true|
|Max sentense length:|512|

## Data Source

[https://huggingface.co/jplu/tf-xlm-r-ner-40-lang](https://huggingface.co/jplu/tf-xlm-r-ner-40-lang)

## Benchmarking

```bash
## Metrics on evaluation set:
### Average over the 40 languages
Number of documents: 262300

```
precision    recall  f1-score   support

ORG       0.81      0.81      0.81    102452
PER       0.90      0.91      0.91    108978
LOC       0.86      0.89      0.87    121868

micro avg       0.86      0.87      0.87    333298
macro avg       0.86      0.87      0.87    333298
```

### Afrikaans
Number of documents: 1000
```
precision    recall  f1-score   support

ORG       0.89      0.88      0.88       582
PER       0.89      0.97      0.93       369
LOC       0.84      0.90      0.86       518

micro avg       0.87      0.91      0.89      1469
macro avg       0.87      0.91      0.89      1469
``` 

### Arabic
Number of documents: 10000
```
precision    recall  f1-score   support

ORG       0.83      0.84      0.84      3507
PER       0.90      0.91      0.91      3643
LOC       0.88      0.89      0.88      3604

micro avg       0.87      0.88      0.88     10754
macro avg       0.87      0.88      0.88     10754
```

### Basque
Number of documents: 10000
```
precision    recall  f1-score   support

LOC       0.88      0.93      0.91      5228
ORG       0.86      0.81      0.83      3654
PER       0.91      0.91      0.91      4072

micro avg       0.89      0.89      0.89     12954
macro avg       0.89      0.89      0.89     12954
```

### Bengali
Number of documents: 1000
```
precision    recall  f1-score   support

ORG       0.86      0.89      0.87       325
LOC       0.91      0.91      0.91       406
PER       0.96      0.95      0.95       364

micro avg       0.91      0.92      0.91      1095
macro avg       0.91      0.92      0.91      1095
```

### Bulgarian
Number of documents: 1000
```
precision    recall  f1-score   support

ORG       0.86      0.83      0.84      3661
PER       0.92      0.95      0.94      4006
LOC       0.92      0.95      0.94      6449

micro avg       0.91      0.92      0.91     14116
macro avg       0.91      0.92      0.91     14116
```

### Burmese
Number of documents: 100
```
precision    recall  f1-score   support

LOC       0.60      0.86      0.71        37
ORG       0.68      0.63      0.66        30
PER       0.44      0.44      0.44        36

micro avg       0.57      0.65      0.61       103
macro avg       0.57      0.65      0.60       103
```

### Chinese
Number of documents: 10000
```
precision    recall  f1-score   support

ORG       0.70      0.69      0.70      4022
LOC       0.76      0.81      0.78      3830
PER       0.84      0.84      0.84      3706

micro avg       0.76      0.78      0.77     11558
macro avg       0.76      0.78      0.77     11558
```

### Dutch
Number of documents: 10000
```
precision    recall  f1-score   support

ORG       0.87      0.87      0.87      3930
PER       0.95      0.95      0.95      4377
LOC       0.91      0.92      0.91      4813

micro avg       0.91      0.92      0.91     13120
macro avg       0.91      0.92      0.91     13120
```

### English
Number of documents: 10000
```
precision    recall  f1-score   support

LOC       0.83      0.84      0.84      4781
PER       0.89      0.90      0.89      4559
ORG       0.75      0.75      0.75      4633

micro avg       0.82      0.83      0.83     13973
macro avg       0.82      0.83      0.83     13973
```

### Estonian
Number of documents: 10000
```
precision    recall  f1-score   support

LOC       0.89      0.92      0.91      5654
ORG       0.85      0.85      0.85      3878
PER       0.94      0.94      0.94      4026

micro avg       0.90      0.91      0.90     13558
macro avg       0.90      0.91      0.90     13558
```

### Finnish
Number of documents: 10000
```
precision    recall  f1-score   support

ORG       0.84      0.83      0.84      4104
LOC       0.88      0.90      0.89      5307
PER       0.95      0.94      0.94      4519

micro avg       0.89      0.89      0.89     13930
macro avg       0.89      0.89      0.89     13930
```

### French
Number of documents: 10000
```
precision    recall  f1-score   support

LOC       0.90      0.89      0.89      4808
ORG       0.84      0.87      0.85      3876
PER       0.94      0.93      0.94      4249

micro avg       0.89      0.90      0.90     12933
macro avg       0.89      0.90      0.90     12933
```

### Georgian
Number of documents: 10000
```
precision    recall  f1-score   support

PER       0.90      0.91      0.90      3964
ORG       0.83      0.77      0.80      3757
LOC       0.82      0.88      0.85      4894

micro avg       0.84      0.86      0.85     12615
macro avg       0.84      0.86      0.85     12615
```

### German
Number of documents: 10000
```
precision    recall  f1-score   support

LOC       0.85      0.90      0.87      4939
PER       0.94      0.91      0.92      4452
ORG       0.79      0.78      0.79      4247

micro avg       0.86      0.86      0.86     13638
macro avg       0.86      0.86      0.86     13638
```

### Greek
Number of documents: 10000
```
precision    recall  f1-score   support

ORG       0.86      0.85      0.85      3771
LOC       0.88      0.91      0.90      4436
PER       0.91      0.93      0.92      3894

micro avg       0.88      0.90      0.89     12101
macro avg       0.88      0.90      0.89     12101
```

### Hebrew
Number of documents: 10000
```
precision    recall  f1-score   support

PER       0.87      0.88      0.87      4206
ORG       0.76      0.75      0.76      4190
LOC       0.85      0.85      0.85      4538

micro avg       0.83      0.83      0.83     12934
macro avg       0.82      0.83      0.83     12934
```

### Hindi
Number of documents: 1000
```
precision    recall  f1-score   support

ORG       0.78      0.81      0.79       362
LOC       0.83      0.85      0.84       422
PER       0.90      0.95      0.92       427

micro avg       0.84      0.87      0.85      1211
macro avg       0.84      0.87      0.85      1211
```

### Hungarian
Number of documents: 10000
```
precision    recall  f1-score   support

PER       0.95      0.95      0.95      4347
ORG       0.87      0.88      0.87      3988
LOC       0.90      0.92      0.91      5544

micro avg       0.91      0.92      0.91     13879
macro avg       0.91      0.92      0.91     13879
```

### Indonesian
Number of documents: 10000
```
precision    recall  f1-score   support

ORG       0.88      0.89      0.88      3735
LOC       0.93      0.95      0.94      3694
PER       0.93      0.93      0.93      3947

micro avg       0.91      0.92      0.92     11376
macro avg       0.91      0.92      0.92     11376
```

### Italian
Number of documents: 10000
```
precision    recall  f1-score   support

LOC       0.88      0.88      0.88      4592
ORG       0.86      0.86      0.86      4088
PER       0.96      0.96      0.96      4732

micro avg       0.90      0.90      0.90     13412
macro avg       0.90      0.90      0.90     13412
```

### Japanese
Number of documents: 10000
```
precision    recall  f1-score   support

ORG       0.62      0.61      0.62      4184
PER       0.76      0.81      0.78      3812
LOC       0.68      0.74      0.71      4281

micro avg       0.69      0.72      0.70     12277
macro avg       0.69      0.72      0.70     12277
```

### Javanese
Number of documents: 100
```
precision    recall  f1-score   support

ORG       0.79      0.80      0.80        46
PER       0.81      0.96      0.88        26
LOC       0.75      0.75      0.75        40

micro avg       0.78      0.82      0.80       112
macro avg       0.78      0.82      0.80       112
```

### Kazakh
Number of documents: 1000
```
precision    recall  f1-score   support

ORG       0.76      0.61      0.68       307
LOC       0.78      0.90      0.84       461
PER       0.87      0.91      0.89       367

micro avg       0.81      0.83      0.82      1135
macro avg       0.81      0.83      0.81      1135
```

### Korean
Number of documents: 10000
```
precision    recall  f1-score   support

LOC       0.86      0.89      0.88      5097
ORG       0.79      0.74      0.77      4218
PER       0.83      0.86      0.84      4014

micro avg       0.83      0.83      0.83     13329
macro avg       0.83      0.83      0.83     13329
```

### Malay
Number of documents: 1000
```
precision    recall  f1-score   support

ORG       0.87      0.89      0.88       368
PER       0.92      0.91      0.91       366
LOC       0.94      0.95      0.95       354

micro avg       0.91      0.92      0.91      1088
macro avg       0.91      0.92      0.91      1088
```

### Malayalam
Number of documents: 1000
```
precision    recall  f1-score   support

ORG       0.75      0.74      0.75       347
PER       0.84      0.89      0.86       417
LOC       0.74      0.75      0.75       391

micro avg       0.78      0.80      0.79      1155
macro avg       0.78      0.80      0.79      1155
```

### Marathi
Number of documents: 1000
```
precision    recall  f1-score   support

PER       0.89      0.94      0.92       394
LOC       0.82      0.84      0.83       457
ORG       0.84      0.78      0.81       339

micro avg       0.85      0.86      0.85      1190
macro avg       0.85      0.86      0.85      1190
```

### Persian
Number of documents: 10000
```
precision    recall  f1-score   support

PER       0.93      0.92      0.93      3540
LOC       0.93      0.93      0.93      3584
ORG       0.89      0.92      0.90      3370

micro avg       0.92      0.92      0.92     10494
macro avg       0.92      0.92      0.92     10494
```

### Portuguese
Number of documents: 10000
```
precision    recall  f1-score   support

LOC       0.90      0.91      0.91      4819
PER       0.94      0.92      0.93      4184
ORG       0.84      0.88      0.86      3670

micro avg       0.89      0.91      0.90     12673
macro avg       0.90      0.91      0.90     12673
```

### Russian
Number of documents: 10000
```
precision    recall  f1-score   support

PER       0.93      0.96      0.95      3574
LOC       0.87      0.89      0.88      4619
ORG       0.82      0.80      0.81      3858

micro avg       0.87      0.88      0.88     12051
macro avg       0.87      0.88      0.88     12051
```

### Spanish
Number of documents: 10000
```
precision    recall  f1-score   support

PER       0.95      0.93      0.94      3891
ORG       0.86      0.88      0.87      3709
LOC       0.89      0.91      0.90      4553

micro avg       0.90      0.91      0.90     12153
macro avg       0.90      0.91      0.90     12153
```

### Swahili
Number of documents: 1000
```
precision    recall  f1-score   support

ORG       0.82      0.85      0.83       349
PER       0.95      0.92      0.94       403
LOC       0.86      0.89      0.88       450

micro avg       0.88      0.89      0.88      1202
macro avg       0.88      0.89      0.88      1202
```

### Tagalog
Number of documents: 1000
```
precision    recall  f1-score   support

LOC       0.90      0.91      0.90       338
ORG       0.83      0.91      0.87       339
PER       0.96      0.93      0.95       350

micro avg       0.90      0.92      0.91      1027
macro avg       0.90      0.92      0.91      1027
```

### Tamil
Number of documents: 1000
```
precision    recall  f1-score   support

PER       0.90      0.92      0.91       392
ORG       0.77      0.76      0.76       370
LOC       0.78      0.81      0.79       421

micro avg       0.82      0.83      0.82      1183
macro avg       0.82      0.83      0.82      1183
```

### Telugu
Number of documents: 1000
```
precision    recall  f1-score   support

ORG       0.67      0.55      0.61       347
LOC       0.78      0.87      0.82       453
PER       0.73      0.86      0.79       393

micro avg       0.74      0.77      0.76      1193
macro avg       0.73      0.77      0.75      1193
```

### Thai
Number of documents: 10000
```
precision    recall  f1-score   support

LOC       0.63      0.76      0.69      3928
PER       0.78      0.83      0.80      6537
ORG       0.59      0.59      0.59      4257

micro avg       0.68      0.74      0.71     14722
macro avg       0.68      0.74      0.71     14722
```

### Turkish
Number of documents: 10000
```
precision    recall  f1-score   support

PER       0.94      0.94      0.94      4337
ORG       0.88      0.89      0.88      4094
LOC       0.90      0.92      0.91      4929

micro avg       0.90      0.92      0.91     13360
macro avg       0.91      0.92      0.91     13360
```

### Urdu
Number of documents: 1000
```
precision    recall  f1-score   support

LOC       0.90      0.95      0.93       352
PER       0.96      0.96      0.96       333
ORG       0.91      0.90      0.90       326

micro avg       0.92      0.94      0.93      1011
macro avg       0.92      0.94      0.93      1011
```

### Vietnamese
Number of documents: 10000
```
precision    recall  f1-score   support

ORG       0.86      0.87      0.86      3579
LOC       0.88      0.91      0.90      3811
PER       0.92      0.93      0.93      3717

micro avg       0.89      0.90      0.90     11107
macro avg       0.89      0.90      0.90     11107
```

### Yoruba
Number of documents: 100
```
precision    recall  f1-score   support

LOC       0.54      0.72      0.62        36
ORG       0.58      0.31      0.41        35
PER       0.77      1.00      0.87        36

micro avg       0.64      0.68      0.66       107
macro avg       0.63      0.68      0.63       107

```
