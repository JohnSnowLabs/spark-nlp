---
layout: model
title: Social Determinants of Health (slim)
author: John Snow Labs
name: sdoh_slim_wip
date: 2022-11-15
tags: [en, licensed, sdoh, social_determinants, ner, clinical, public_health]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.2.1
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model extracts terminology-related entities to `Social Determinants of Health ` from various kinds of biomedical documents.

## Predicted Entities

`Housing`, `Substance_Frequency`, `Childhood_Development`, `race_ethnicity`, `mental_health`, `Other_Disease`, `age`, `smoking`, `Disability`, `gender`, `diet`, `Substance_Quantity`, `employment`, `Family_Member`, `marital_status`, `Geographic_Entity`, `Sexual_Orientation`, `alcohol`, `Substance_Use`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sdoh_slim_wip_en_4.2.1_3.0_1668513558754.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")
        
sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
        .setInputCols(["document"])\
        .setOutputCol("sentence")

tokenizer = Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

clinical_embeddings = WordEmbeddingsModel.pretrained('embeddings_clinical', "en", "clinical/models")\
        .setInputCols(["sentence", "token"])\
        .setOutputCol("embeddings")

ner_model = MedicalNerModel.pretrained("sdoh_slim_wip", "en", "clinical/models")\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")

ner_converter = NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        clinical_embeddings,
        ner_model,
        ner_converter]

text = [""" Mother states that he does smoke, there is a family hx of alcohol on both maternal and paternal sides of the family, maternal grandfather who died of alcohol related complications and paternal grandmother with severe alcoholism. Pts own drinking began at age 16, living in LA, had a DUI at age 17 after totaling a new car that his mother bought for him, he was married. """]
data = spark.createDataFrame([text]).toDF("text")

result = nlpPipelineçfit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
    .setInputCols("document")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val ner_model = BertForTokenClassification.pretrained("sdoh_slim_wipe" "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("ner")
    .setCaseSensitive(True)
    .setMaxSentenceLength(512)

val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val pipeline = new PipelineModel().setStages(Array(document_assembler, 
                                                   sentence_detector,
                                                   tokenizer,
                                                   ner_model,
                                                   ner_converter))

val data = Seq("""Mother states that there is a family hx of alcohol on both maternal and paternal sides of the family, maternal grandfather who died of alcohol related complications and paternal grandmother with severe alcoholism. Pts own drinking began at age 16, had a DUI at age 17 after totaling a new car that his mother bought for him, he was married.""").toDS.toDF("text")

val result = model.fit(data).transform(data)
```
</div>

## Results

```bash
+-------------+-------------------+
|        token|          ner_label|
+-------------+-------------------+
|       Mother|    B-Family_Member|
|       states|                  O|
|         that|                  O|
|           he|           B-gender|
|         does|                  O|
|        smoke|          B-smoking|
|            ,|                  O|
|        there|                  O|
|           is|                  O|
|            a|                  O|
|       family|                  O|
|           hx|                  O|
|           of|                  O|
|      alcohol|          B-alcohol|
|           on|                  O|
|         both|                  O|
|     maternal|    B-Family_Member|
|          and|                  O|
|     paternal|    B-Family_Member|
|        sides|                  O|
|           of|                  O|
|          the|                  O|
|       family|                  O|
|            ,|                  O|
|     maternal|    B-Family_Member|
|  grandfather|    B-Family_Member|
|          who|                  O|
|         died|                  O|
|           of|                  O|
|      alcohol|          B-alcohol|
|      related|                  O|
|complications|                  O|
|          and|                  O|
|     paternal|    B-Family_Member|
|  grandmother|    B-Family_Member|
|         with|                  O|
|       severe|          B-alcohol|
|   alcoholism|          I-alcohol|
|            .|                  O|
|          Pts|                  O|
|          own|                  O|
|     drinking|          B-alcohol|
|        began|                  O|
|           at|                  O|
|          age|              B-age|
|           16|              I-age|
|            ,|                  O|
|       living|                  O|
|           in|                  O|
|           LA|B-Geographic_Entity|
|            ,|                  O|
|          had|                  O|
|            a|                  O|
|          DUI|                  O|
|           at|                  O|
|          age|              B-age|
|           17|              I-age|
|        after|                  O|
|     totaling|                  O|
|            a|                  O|
|          new|                  O|
|          car|                  O|
|         that|                  O|
|          his|           B-gender|
|       mother|    B-Family_Member|
|       bought|                  O|
|          for|                  O|
|          him|           B-gender|
|            ,|                  O|
|           he|           B-gender|
|          was|                  O|
|      married|   B-marital_status|
+-------------+-------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sdoh_slim_wip|
|Compatibility:|Healthcare NLP 4.2.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|2.8 MB|

## References

Manuel annotations have been made over [MTSamples](https://mtsamples.com/) and [MIMIC ](https://physionet.org/content/mimiciii/1.4/) datasets.

## Benchmarking

```bash
                 label      precision  recall  f1-score   support
B-Childhood_Development       1.00      1.00      1.00         1
           B-Disability       0.91      0.93      0.92        57
        B-Family_Member       0.93      0.98      0.95      2412
    B-Geographic_Entity       0.77      0.79      0.78        82
              B-Housing       0.82      0.63      0.71       183
        B-Other_Disease       0.79      0.81      0.80       381
   B-Sexual_Orientation       0.75      0.90      0.82        10
  B-Substance_Frequency       0.89      0.90      0.89        88
   B-Substance_Quantity       0.79      0.79      0.79        72
        B-Substance_Use       0.79      0.82      0.81       213
                  B-age       0.90      0.92      0.91       277
              B-alcohol       0.92      0.89      0.90       410
                 B-diet       1.00      1.00      1.00         6
           B-employment       0.86      0.84      0.85      1926
               B-gender       0.97      0.99      0.98      6161
       B-marital_status       0.94      0.92      0.93       184
        B-mental_health       0.83      0.74      0.78       487
       B-race_ethnicity       0.94      0.88      0.91        34
              B-smoking       0.93      0.96      0.95       209
I-Childhood_Development       1.00      1.00      1.00         3
           I-Disability       0.73      0.53      0.62        15
        I-Family_Member       0.85      0.86      0.85       138
    I-Geographic_Entity       1.00      0.85      0.92        13
              I-Housing       0.88      0.70      0.78       362
        I-Other_Disease       0.78      0.77      0.78       256
  I-Substance_Frequency       0.86      0.83      0.84        75
   I-Substance_Quantity       0.87      0.84      0.86       174
        I-Substance_Use       0.87      0.82      0.84       182
                  I-age       0.89      0.96      0.92       589
              I-alcohol       0.81      0.76      0.78       159
                 I-diet       1.00      1.00      1.00         9
           I-employment       0.58      0.75      0.65       369
               I-gender       0.56      0.94      0.70       231
       I-marital_status       0.75      0.55      0.63        11
        I-mental_health       0.77      0.56      0.65       241
       I-race_ethnicity       1.00      1.00      1.00        15
              I-smoking       0.86      0.93      0.90        46
                      O       0.99      0.99      0.99    148829
               accuracy         -         -       0.98    164910
              macro_avg       0.86      0.85      0.85    164910
           weighted_avg       0.98      0.98      0.98    164910
```
