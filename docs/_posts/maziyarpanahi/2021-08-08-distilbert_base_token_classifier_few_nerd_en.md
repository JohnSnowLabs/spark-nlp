---
layout: model
title: DistilBERT Token Classification -  Few-NERD (distilbert_base_token_classifier_few_nerd)
author: John Snow Labs
name: distilbert_base_token_classifier_few_nerd
date: 2021-08-08
tags: [token_classification, ner, distilbert, few_nerd, open_source, en, english]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.2.0
spark_version: 2.4
supported: true
annotator: DistilBertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

`DistilBERT Model` with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_base_token_classifier_few_nerd_en_3.2.0_2.4_1628435975886.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = DistilBertForTokenClassification \
      .pretrained('distilbert_base_token_classifier_few_nerd', 'en') \
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

val tokenClassifier = DistilBertForTokenClassification.pretrained("distilbert_base_token_classifier_few_nerd", "en")
      .setInputCols("document", "token")
      .setOutputCol("ner")
      .setCaseSensitive(true)
      .setMaxSentenceLength(512)

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, tokenClassifier))

val example = Seq.empty["My name is John!"].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_base_token_classifier_few_nerd|
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


                                   label  precision    recall  f1-score   support
                                       O       0.98      0.98      0.98    365750
                    art-broadcastprogram       0.66      0.66      0.66       890
                                art-film       0.76      0.74      0.75      1039
                               art-music       0.89      0.79      0.84      1773
                               art-other       0.39      0.41      0.40       729
                            art-painting       0.48      0.46      0.47        91
                          art-writtenart       0.68      0.72      0.70      1570
                        building-airport       0.84      0.88      0.86       391
                       building-hospital       0.79      0.89      0.84       577
                          building-hotel       0.85      0.80      0.83       526
                        building-library       0.83      0.87      0.85       715
                          building-other       0.64      0.67      0.66      3448
                     building-restaurant       0.63      0.52      0.57       283
                 building-sportsfacility       0.63      0.80      0.71       495
                        building-theater       0.77      0.85      0.81       529
event-attack/battle/war/militaryconflict       0.82      0.87      0.84      1583
                          event-disaster       0.73      0.71      0.72       317
                          event-election       0.64      0.46      0.53       282
                             event-other       0.64      0.61      0.62      1634
                           event-protest       0.42      0.33      0.37       227
                       event-sportsevent       0.73      0.78      0.75      1975
                            location-GPE       0.82      0.86      0.84     13112
                  location-bodiesofwater       0.84      0.82      0.83      1210
                         location-island       0.81      0.80      0.81       666
                       location-mountain       0.83      0.78      0.80       734
                          location-other       0.43      0.37      0.40      2207
                           location-park       0.72      0.80      0.76       634
   location-road/railway/highway/transit       0.77      0.79      0.78      1861
                    organization-company       0.71      0.77      0.74      3982
                  organization-education       0.87      0.88      0.87      3432
organization-government/governmentagency       0.63      0.56      0.59      2178
            organization-media/newspaper       0.63      0.64      0.63      1291
                      organization-other       0.62      0.64      0.63      5989
             organization-politicalparty       0.75      0.79      0.77      1199
                   organization-religion       0.65      0.72      0.68       830
           organization-showorganization       0.71      0.75      0.73       933
               organization-sportsleague       0.74      0.59      0.66      1088
                 organization-sportsteam       0.79      0.81      0.80      2374
                    other-astronomything       0.80      0.80      0.80       625
                             other-award       0.81      0.72      0.77      1873
                      other-biologything       0.70      0.68      0.69      1282
                     other-chemicalthing       0.70      0.56      0.62       881
                          other-currency       0.74      0.81      0.78       608
                           other-disease       0.71      0.71      0.71       825
                 other-educationaldegree       0.72      0.79      0.75       599
                               other-god       0.68      0.61      0.64       316
                          other-language       0.75      0.82      0.78       539
                               other-law       0.83      0.81      0.82       966
                       other-livingthing       0.62      0.70      0.66       696
                           other-medical       0.59      0.47      0.52       293
                            person-actor       0.84      0.80      0.82      1510
                    person-artist/author       0.73      0.77      0.75      3083
                          person-athlete       0.83      0.84      0.84      2519
                         person-director       0.75      0.69      0.72       535
                            person-other       0.69      0.68      0.68      7601
                       person-politician       0.70      0.72      0.71      2588
                          person-scholar       0.57      0.56      0.56       657
                          person-soldier       0.64      0.65      0.65       573
                        product-airplane       0.79      0.68      0.73       781
                             product-car       0.81      0.77      0.79       779
                            product-food       0.55      0.52      0.53       345
                            product-game       0.74      0.80      0.77       534
                           product-other       0.59      0.44      0.51      1751
                            product-ship       0.69      0.76      0.72       333
                        product-software       0.64      0.61      0.62       693
                           product-train       0.54      0.69      0.61       274
                          product-weapon       0.74      0.68      0.71       611
                                accuracy          -         -      0.93    463214
                               macro-avg       0.71      0.71      0.71    463214
                            weighted-avg       0.93      0.93      0.93    463214



processed 463214 tokens with 48764 phrases; found: 50982 phrases; correct: 33677.
accuracy:  72.96%; (non-O)
accuracy:  92.73%; precision:  66.06%; recall:  69.06%; FB1:  67.53
              GPE: precision:  78.80%; recall:  84.16%; FB1:  81.39  11040
            actor: precision:  79.59%; recall:  76.83%; FB1:  78.18  779
         airplane: precision:  68.03%; recall:  52.22%; FB1:  59.08  294
          airport: precision:  77.70%; recall:  80.99%; FB1:  79.31  148
    artist/author: precision:  68.07%; recall:  73.99%; FB1:  70.91  1876
   astronomything: precision:  71.43%; recall:  73.86%; FB1:  72.63  364
          athlete: precision:  79.02%; recall:  82.86%; FB1:  80.90  1554
attack/battle/war/militaryconflict: precision:  69.39%; recall:  80.63%; FB1:  74.59  624
            award: precision:  60.16%; recall:  58.51%; FB1:  59.33  497
     biologything: precision:  61.11%; recall:  62.79%; FB1:  61.94  900
    bodiesofwater: precision:  77.96%; recall:  77.32%; FB1:  77.64  608
 broadcastprogram: precision:  57.76%; recall:  61.28%; FB1:  59.47  348
              car: precision:  68.98%; recall:  69.35%; FB1:  69.17  374
    chemicalthing: precision:  56.49%; recall:  49.82%; FB1:  52.94  478
          company: precision:  62.69%; recall:  68.48%; FB1:  65.45  2093
         currency: precision:  66.37%; recall:  72.06%; FB1:  69.10  443
         director: precision:  68.08%; recall:  63.21%; FB1:  65.56  260
         disaster: precision:  50.00%; recall:  56.59%; FB1:  53.09  146
          disease: precision:  61.30%; recall:  65.40%; FB1:  63.28  478
        education: precision:  77.39%; recall:  79.55%; FB1:  78.45  1141
educationaldegree: precision:  50.93%; recall:  56.77%; FB1:  53.69  214
         election: precision:  27.40%; recall:  24.10%; FB1:  25.64  73
             film: precision:  71.21%; recall:  68.40%; FB1:  69.77  389
             food: precision:  45.05%; recall:  41.62%; FB1:  43.27  182
             game: precision:  61.90%; recall:  72.96%; FB1:  66.98  231
              god: precision:  63.36%; recall:  65.10%; FB1:  64.22  262
government/governmentagency: precision:  46.50%; recall:  41.59%; FB1:  43.91  686
         hospital: precision:  69.63%; recall:  79.17%; FB1:  74.09  191
            hotel: precision:  65.93%; recall:  65.57%; FB1:  65.75  182
           island: precision:  71.27%; recall:  71.47%; FB1:  71.37  362
         language: precision:  69.28%; recall:  79.86%; FB1:  74.19  498
              law: precision:  55.17%; recall:  60.61%; FB1:  57.76  290
          library: precision:  65.50%; recall:  73.89%; FB1:  69.44  229
      livingthing: precision:  56.56%; recall:  62.69%; FB1:  59.47  511
  media/newspaper: precision:  53.78%; recall:  62.01%; FB1:  57.60  701
          medical: precision:  58.10%; recall:  55.71%; FB1:  56.88  210
         mountain: precision:  73.03%; recall:  70.84%; FB1:  71.92  356
            music: precision:  76.67%; recall:  73.61%; FB1:  75.11  553
            other: precision:  57.66%; recall:  58.75%; FB1:  58.20  10723
         painting: precision:  33.33%; recall:  40.00%; FB1:  36.36  30
             park: precision:  60.87%; recall:  71.30%; FB1:  65.67  253
   politicalparty: precision:  60.38%; recall:  69.53%; FB1:  64.63  631
       politician: precision:  64.91%; recall:  66.71%; FB1:  65.80  1522
          protest: precision:  26.14%; recall:  26.14%; FB1:  26.14  88
         religion: precision:  49.78%; recall:  56.34%; FB1:  52.86  464
       restaurant: precision:  49.54%; recall:  42.52%; FB1:  45.76  109
road/railway/highway/transit: precision:  66.63%; recall:  71.47%; FB1:  68.96  827
          scholar: precision:  51.22%; recall:  51.50%; FB1:  51.36  369
             ship: precision:  58.85%; recall:  67.96%; FB1:  63.08  209
 showorganization: precision:  58.58%; recall:  65.47%; FB1:  61.83  466
         software: precision:  56.30%; recall:  58.91%; FB1:  57.58  405
          soldier: precision:  55.24%; recall:  58.91%; FB1:  57.02  353
      sportsevent: precision:  53.74%; recall:  62.46%; FB1:  57.77  802
   sportsfacility: precision:  59.92%; recall:  75.50%; FB1:  66.81  252
     sportsleague: precision:  61.78%; recall:  55.87%; FB1:  58.68  416
       sportsteam: precision:  70.50%; recall:  76.65%; FB1:  73.45  1332
          theater: precision:  64.32%; recall:  74.11%; FB1:  68.87  227
            train: precision:  40.25%; recall:  54.70%; FB1:  46.38  159
           weapon: precision:  59.41%; recall:  52.96%; FB1:  56.00  271
       writtenart: precision:  53.83%; recall:  59.44%; FB1:  56.49  509
```