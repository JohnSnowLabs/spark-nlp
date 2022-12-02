---
layout: model
title: BERT Token Classification -  Few-NERD (bert_base_token_classifier_few_nerd)
author: John Snow Labs
name: bert_base_token_classifier_few_nerd
date: 2021-08-08
tags: [few_nerd, ner, open_source, en, english, bert, token_classification]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.2.0
spark_version: 2.4
supported: true
annotator: BertForTokenClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

`BERT Model` with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

This model is fine-tuned on the Few-NERD dataset. Few-NERD is a large-scale, fine-grained manually annotated named entity recognition dataset, which contains 8 coarse-grained types, 66 fine-grained types, 188,200 sentences, 491,711 entities, and 4,601,223 tokens. Three benchmark tasks are built, one is supervised (Few-NERD (SUP)) and the other two are few-shot (Few-NERD (INTRA) and Few-NERD (INTER)). Few-NERD is collected by researchers from Tsinghua University and DAMO Academy, Alibaba Group.

## Predicted Entities

- art-broadcastprogram
- art-film
- art-music
- art-other
- art-painting
- art-writtenart
- building-airport
- building-hospital
- building-hotel
- building-library
- building-other
- building-restaurant
- building-sportsfacility
- building-theater
- event-attack/battle/war/militaryconflict
- event-disaster
- event-election
- event-other
- event-protest
- event-sportsevent
- location-GPE
- location-bodiesofwater
- location-island
- location-mountain
- location-other
- location-park
- location-road/railway/highway/transit
- organization-company
- organization-education
- organization-government/governmentagency
- organization-media/newspaper
- organization-other
- organization-politicalparty
- organization-religion
- organization-showorganization
- organization-sportsleague
- organization-sportsteam
- other-astronomything
- other-award
- other-biologything
- other-chemicalthing
- other-currency
- other-disease
- other-educationaldegree
- other-god
- other-language
- other-law
- other-livingthing
- other-medical
- person-actor
- person-artist/author
- person-athlete
- person-director
- person-other
- person-politician
- person-scholar
- person-soldier
- product-airplane
- product-car
- product-food
- product-game
- product-other
- product-ship
- product-software
- product-train
- product-weapon

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_token_classifier_few_nerd_en_3.2.0_2.4_1628433381711.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = BertForTokenClassification \
.pretrained('bert_base_token_classifier_few_nerd', 'en') \
.setInputCols(['token', 'document']) \
.setOutputCol('ner') \
.setCaseSensitive(True) \
.setMaxSentenceLength(512)

pipeline = Pipeline(stages=[
document_assembler, 
tokenizer,
tokenClassifier
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

val tokenClassifier = BertForTokenClassification.pretrained("bert_base_token_classifier_few_nerd", "en")
.setInputCols("document", "token")
.setOutputCol("ner")
.setCaseSensitive(true)
.setMaxSentenceLength(512)

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, tokenClassifier))

val example = Seq.empty["My name is John!"].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.token_bert.few_nerd").predict("""My name is John!""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_token_classifier_few_nerd|
|Compatibility:|Spark NLP 3.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[ner]|
|Language:|en|
|Case sensitive:|true|
|Max sentense length:|512|

## Data Source

[https://github.com/thunlp/Few-NERD](https://github.com/thunlp/Few-NERD)

## Benchmarking

```bash
Test:

precision    recall  f1-score   support

O       0.98      0.98      0.98    365750
art-broadcastprogram       0.66      0.66      0.66       890
art-film       0.78      0.78      0.78      1039
art-music       0.85      0.81      0.83      1773
art-other       0.40      0.40      0.40       729
art-painting       0.51      0.43      0.47        91
art-writtenart       0.69      0.70      0.70      1570
building-airport       0.83      0.88      0.85       391
building-hospital       0.80      0.89      0.84       577
building-hotel       0.87      0.80      0.83       526
building-library       0.81      0.86      0.83       715
building-other       0.64      0.67      0.65      3448
building-restaurant       0.72      0.57      0.64       283
building-sportsfacility       0.65      0.82      0.72       495
building-theater       0.78      0.90      0.83       529
event-attack/battle/war/militaryconflict       0.82      0.87      0.85      1583
event-disaster       0.67      0.73      0.70       317
event-election       0.56      0.46      0.51       282
event-other       0.65      0.57      0.60      1634
event-protest       0.41      0.48      0.44       227
event-sportsevent       0.74      0.80      0.77      1975
location-GPE       0.82      0.86      0.84     13112
location-bodiesofwater       0.83      0.82      0.83      1210
location-island       0.81      0.81      0.81       666
location-mountain       0.82      0.78      0.80       734
location-other       0.45      0.36      0.40      2207
location-park       0.71      0.81      0.76       634
location-road/railway/highway/transit       0.76      0.79      0.77      1861
organization-company       0.75      0.77      0.76      3982
organization-education       0.87      0.88      0.88      3432
organization-government/governmentagency       0.65      0.60      0.62      2178
organization-media/newspaper       0.63      0.67      0.65      1291
organization-other       0.63      0.64      0.64      5989
organization-politicalparty       0.75      0.81      0.78      1199
organization-religion       0.65      0.74      0.69       830
organization-showorganization       0.74      0.78      0.76       933
organization-sportsleague       0.75      0.60      0.67      1088
organization-sportsteam       0.79      0.84      0.81      2374
other-astronomything       0.80      0.82      0.81       625
other-award       0.80      0.73      0.77      1873
other-biologything       0.69      0.70      0.69      1282
other-chemicalthing       0.70      0.56      0.62       881
other-currency       0.75      0.85      0.80       608
other-disease       0.71      0.73      0.72       825
other-educationaldegree       0.73      0.80      0.76       599
other-god       0.70      0.67      0.69       316
other-language       0.75      0.83      0.78       539
other-law       0.82      0.82      0.82       966
other-livingthing       0.64      0.71      0.67       696
other-medical       0.53      0.45      0.49       293
person-actor       0.85      0.82      0.83      1510
person-artist/author       0.74      0.77      0.76      3083
person-athlete       0.84      0.86      0.85      2519
person-director       0.73      0.73      0.73       535
person-other       0.71      0.68      0.70      7601
person-politician       0.72      0.72      0.72      2588
person-scholar       0.54      0.59      0.56       657
person-soldier       0.63      0.67      0.65       573
product-airplane       0.79      0.69      0.73       781
product-car       0.84      0.79      0.81       779
product-food       0.53      0.56      0.54       345
product-game       0.81      0.81      0.81       534
product-other       0.60      0.45      0.51      1751
product-ship       0.65      0.71      0.68       333
product-software       0.62      0.66      0.64       693
product-train       0.50      0.72      0.59       274
product-weapon       0.74      0.70      0.72       611

accuracy                           0.93    463214
macro avg       0.71      0.72      0.71    463214
weighted avg       0.93      0.93      0.93    463214



processed 463214 tokens with 48764 phrases; found: 51017 phrases; correct: 34149.
accuracy:  73.78%; (non-O)
accuracy:  92.88%; precision:  66.94%; recall:  70.03%; FB1:  68.45
GPE: precision:  79.57%; recall:  84.68%; FB1:  82.05  11001
actor: precision:  81.64%; recall:  78.81%; FB1:  80.20  779
airplane: precision:  65.69%; recall:  52.48%; FB1:  58.35  306
airport: precision:  74.17%; recall:  78.87%; FB1:  76.45  151
artist/author: precision:  69.20%; recall:  74.45%; FB1:  71.73  1857
astronomything: precision:  70.49%; recall:  73.30%; FB1:  71.87  366
athlete: precision:  80.10%; recall:  83.94%; FB1:  81.98  1553
attack/battle/war/militaryconflict: precision:  72.32%; recall:  81.75%; FB1:  76.75  607
award: precision:  58.38%; recall:  59.30%; FB1:  58.83  519
biologything: precision:  61.19%; recall:  63.36%; FB1:  62.25  907
bodiesofwater: precision:  76.54%; recall:  77.16%; FB1:  76.85  618
broadcastprogram: precision:  57.97%; recall:  60.98%; FB1:  59.44  345
car: precision:  68.66%; recall:  67.74%; FB1:  68.20  367
chemicalthing: precision:  57.74%; recall:  50.92%; FB1:  54.12  478
company: precision:  66.25%; recall:  68.84%; FB1:  67.52  1991
currency: precision:  66.60%; recall:  76.23%; FB1:  71.09  467
director: precision:  68.20%; recall:  68.93%; FB1:  68.56  283
disaster: precision:  46.54%; recall:  57.36%; FB1:  51.39  159
disease: precision:  58.45%; recall:  65.62%; FB1:  61.83  503
education: precision:  77.32%; recall:  80.18%; FB1:  78.73  1151
educationaldegree: precision:  55.30%; recall:  62.50%; FB1:  58.68  217
election: precision:  26.83%; recall:  26.51%; FB1:  26.67  82
film: precision:  73.77%; recall:  74.32%; FB1:  74.05  408
food: precision:  43.35%; recall:  44.67%; FB1:  44.00  203
game: precision:  67.61%; recall:  73.47%; FB1:  70.42  213
god: precision:  67.04%; recall:  70.98%; FB1:  68.95  270
government/governmentagency: precision:  47.22%; recall:  45.37%; FB1:  46.28  737
hospital: precision:  67.01%; recall:  77.38%; FB1:  71.82  194
hotel: precision:  68.93%; recall:  66.67%; FB1:  67.78  177
island: precision:  72.58%; recall:  72.58%; FB1:  72.58  361
language: precision:  68.77%; recall:  80.56%; FB1:  74.20  506
law: precision:  56.51%; recall:  62.50%; FB1:  59.35  292
library: precision:  66.37%; recall:  73.89%; FB1:  69.93  226
livingthing: precision:  59.27%; recall:  63.12%; FB1:  61.13  491
media/newspaper: precision:  52.84%; recall:  62.66%; FB1:  57.34  721
medical: precision:  52.25%; recall:  52.97%; FB1:  52.61  222
mountain: precision:  73.95%; recall:  71.93%; FB1:  72.93  357
music: precision:  76.52%; recall:  74.13%; FB1:  75.31  558
other: precision:  59.20%; recall:  59.14%; FB1:  59.17  10514
painting: precision:  37.04%; recall:  40.00%; FB1:  38.46  27
park: precision:  61.15%; recall:  73.61%; FB1:  66.81  260
politicalparty: precision:  61.72%; recall:  74.45%; FB1:  67.49  661
politician: precision:  66.98%; recall:  68.20%; FB1:  67.58  1508
protest: precision:  28.00%; recall:  39.77%; FB1:  32.86  125
religion: precision:  51.49%; recall:  59.02%; FB1:  55.00  470
restaurant: precision:  60.19%; recall:  51.18%; FB1:  55.32  108
road/railway/highway/transit: precision:  64.51%; recall:  69.78%; FB1:  67.04  834
scholar: precision:  50.13%; recall:  54.50%; FB1:  52.22  399
ship: precision:  49.20%; recall:  50.83%; FB1:  50.00  187
showorganization: precision:  63.75%; recall:  71.70%; FB1:  67.49  469
software: precision:  56.28%; recall:  62.53%; FB1:  59.24  430
soldier: precision:  55.59%; recall:  61.63%; FB1:  58.45  367
sportsevent: precision:  55.30%; recall:  63.48%; FB1:  59.11  792
sportsfacility: precision:  59.68%; recall:  75.50%; FB1:  66.67  253
sportsleague: precision:  65.14%; recall:  58.91%; FB1:  61.87  416
sportsteam: precision:  70.38%; recall:  79.51%; FB1:  74.66  1384
theater: precision:  67.23%; recall:  80.20%; FB1:  73.15  235
train: precision:  39.51%; recall:  54.70%; FB1:  45.88  162
weapon: precision:  55.96%; recall:  50.99%; FB1:  53.36  277
writtenart: precision:  56.65%; recall:  60.95%; FB1:  58.73  496

```