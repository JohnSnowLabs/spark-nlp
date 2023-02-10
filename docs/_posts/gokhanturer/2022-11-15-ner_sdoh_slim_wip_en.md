---
layout: model
title: Social Determinants of Health (slim)
author: John Snow Labs
name: ner_sdoh_slim_wip
date: 2022-11-15
tags: [en, licensed, sdoh, social_determinants, public_health, clinical]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.2.1
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model extracts terminology related to `Social Determinants of Health ` from various kinds of biomedical documents.

## Predicted Entities

`Housing`, `Smoking`, `Substance_Frequency`, `Childhood_Development`, `Age`, `Other_Disease`, `Employment`, `Marital_Status`, `Diet`, `Disability`, `Mental_Health`, `Alcohol`, `Substance_Quantity`, `Family_Member`, `Race_Ethnicity`, `Gender`, `Geographic_Entity`, `Sexual_Orientation`, `Substance_Use`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_sdoh_slim_wip_en_4.2.1_3.0_1668524622964.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_sdoh_slim_wip_en_4.2.1_3.0_1668524622964.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

clinical_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
        .setInputCols(["sentence", "token"])\
        .setOutputCol("embeddings")

ner_model = MedicalNerModel.pretrained("ner_sdoh_slim_wip", "en", "clinical/models")\
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

result = nlpPipeline.fit(data).transform(data)
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

val clinical_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

val ner_model = MedicalNerModel.pretrained("ner_sdoh_slim_wip", "en", "clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"])\

val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val nlpPipeline = new PipelineModel().setStages(Array(document_assembler, 
                                                   sentence_detector,
                                                   tokenizer,
                                                   clinical_embeddings,
                                                   ner_model,
                                                   ner_converter))

val data = Seq("""Mother states that there is a family hx of alcohol on both maternal and paternal sides of the family, maternal grandfather who died of alcohol related complications and paternal grandmother with severe alcoholism. Pts own drinking began at age 16, had a DUI at age 17 after totaling a new car that his mother bought for him, he was married.""").toDS.toDF("text")

val result = nlpPipeline.fit(data).transform(data)
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
|           he|           B-Gender|
|         does|                  O|
|        smoke|          B-Smoking|
|            ,|                  O|
|        there|                  O|
|           is|                  O|
|            a|                  O|
|       family|                  O|
|           hx|                  O|
|           of|                  O|
|      alcohol|          B-Alcohol|
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
|      alcohol|          B-Alcohol|
|      related|                  O|
|complications|                  O|
|          and|                  O|
|     paternal|    B-Family_Member|
|  grandmother|    B-Family_Member|
|         with|                  O|
|       severe|          B-Alcohol|
|   alcoholism|          I-Alcohol|
|            .|                  O|
|          Pts|                  O|
|          own|                  O|
|     drinking|          B-Alcohol|
|        began|                  O|
|           at|                  O|
|          age|              B-Age|
|           16|              I-Age|
|            ,|                  O|
|       living|                  O|
|           in|                  O|
|           LA|B-Geographic_Entity|
|            ,|                  O|
|          had|                  O|
|            a|                  O|
|          DUI|                  O|
|           at|                  O|
|          age|                  O|
|           17|                  O|
|        after|                  O|
|     totaling|                  O|
|            a|                  O|
|          new|                  O|
|          car|                  O|
|         that|                  O|
|          his|           B-Gender|
|       mother|    B-Family_Member|
|       bought|                  O|
|          for|                  O|
|          him|           B-Gender|
|            ,|                  O|
|           he|           B-Gender|
|          was|                  O|
|      married|   B-Marital_Status|
|            .|                  O|
+-------------+-------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_sdoh_slim_wip|
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
                  label    precision   recall   f1-score  support
                  B-Age       0.93      0.90      0.91       277
              B-Alcohol       0.90      0.88      0.89       410
B-Childhood_Development       1.00      1.00      1.00         1
                 B-Diet       1.00      1.00      1.00         6
           B-Disability       0.96      0.95      0.96        57
           B-Employment       0.91      0.79      0.85      1926
        B-Family_Member       0.93      0.97      0.95      2412
               B-Gender       0.97      0.99      0.98      6161
    B-Geographic_Entity       0.81      0.79      0.80        82
              B-Housing       0.82      0.73      0.77       183
       B-Marital_Status       0.93      0.91      0.92       184
        B-Mental_Health       0.85      0.72      0.78       487
        B-Other_Disease       0.77      0.82      0.79       381
       B-Race_Ethnicity       0.91      0.94      0.93        34
   B-Sexual_Orientation       0.75      0.90      0.82        10
              B-Smoking       0.96      0.96      0.96       209
  B-Substance_Frequency       0.92      0.88      0.90        88
   B-Substance_Quantity       0.83      0.79      0.81        72
        B-Substance_Use       0.80      0.82      0.81       213
                  I-Age       0.91      0.95      0.93       589
              I-Alcohol       0.80      0.77      0.79       159
I-Childhood_Development       1.00      1.00      1.00         3
                 I-Diet       1.00      0.89      0.94         9
           I-Disability       1.00      0.53      0.70        15
           I-Employment       0.77      0.62      0.69       369
        I-Family_Member       0.79      0.84      0.81       138
               I-Gender       0.57      0.88      0.69       231
    I-Geographic_Entity       1.00      0.85      0.92        13
              I-Housing       0.86      0.83      0.84       362
       I-Marital_Status       1.00      0.18      0.31        11
        I-Mental_Health       0.81      0.47      0.59       241
        I-Other_Disease       0.75      0.74      0.75       256
       I-Race_Ethnicity       1.00      1.00      1.00        15
              I-Smoking       0.90      0.93      0.91        46
  I-Substance_Frequency       0.85      0.73      0.79        75
   I-Substance_Quantity       0.84      0.88      0.86       174
        I-Substance_Use       0.86      0.84      0.85       182
                      O       0.99      0.99      0.99    148829
               accuracy         -         -       0.98    164910
              macro_avg       0.89      0.83      0.85    164910
           weighted_avg       0.98      0.98      0.98    164910
```
