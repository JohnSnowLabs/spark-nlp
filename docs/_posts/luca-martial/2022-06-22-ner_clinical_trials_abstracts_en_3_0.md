---
layout: model
title: Extract entities in clinical trial abstracts
author: John Snow Labs
name: ner_clinical_trials_abstracts
date: 2022-06-22
tags: [ner, clinical, en, licensed]
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 3.5.3
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This Named Entity Recognition model uses a Deep Learning architecture (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM, CNN.

It extracts relevant entities from clinical trial abstracts. It uses a simplified version of the ontology specified by [Sanchez Graillet, O., et al.](https://pub.uni-bielefeld.de/record/2939477) in order to extract concepts related to trial design, diseases, drugs, population, statistics and publication.

## Predicted Entities

`Age`, `AllocationRatio`, `Author`, `BioAndMedicalUnit`, `CTAnalysisApproach`, `CTDesign`, `Confidence`, `Country`, `DisorderOrSyndrome`, `DoseValue`, `Drug`, `DrugTime`, `Duration`, `Journal`, `NumberPatients`, `PMID`, `PValue`, `PercentagePatients`, `PublicationYear`, `TimePoint`, `Value`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_clinical_trials_abstracts_en_3.5.3_3.0_1655911616789.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")
        
sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")\
        .setInputCols(["document"])\
        .setOutputCol("sentence")

tokenizer = Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical" ,"en", "clinical/models")\
        .setInputCols(["sentence","token"])\
        .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_clinical_trials_abstracts", "en", "clinical/models")\
        .setInputCols(["sentence","token", "word_embeddings"])\
        .setOutputCol("ner")

nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        embeddings,
        clinical_ner])

text = ["A one-year, randomised, multicentre trial comparing insulin glargine with NPH insulin in combination with oral agents in patients with type 2 diabetes."]

df = spark.createDataFrame([text]).toDF("text")

results = nlpPipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")
        .setInputCols("document")
        .setOutputCol("sentence")

val tokenizer = new Tokenizer()
        .setInputCols("sentence")
        .setOutputCol("token")

val embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
        .setInputCols(Array("sentence", "token"))
        .setOutputCol("embeddings")

val clinical_ner = MedicalNerModel.pretrained("ner_clinical_trials_abstracts", "en", "clinical/models")
        .setInputCols(Array("sentence","token","embeddings"))
        .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, embeddings, clinical_ner))

val text = "A one-year, randomised, multicentre trial comparing insulin glargine with NPH insulin in combination with oral agents in patients with type 2 diabetes."

val data = Seq(text).toDF("text")

val results = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-----------+--------------------+
|      token|           ner_label|
+-----------+--------------------+
|          A|                   O|
|   one-year|                   O|
|          ,|                   O|
| randomised|          B-CTDesign|
|          ,|                   O|
|multicentre|          B-CTDesign|
|      trial|                   O|
|  comparing|                   O|
|    insulin|              B-Drug|
|   glargine|              I-Drug|
|       with|                   O|
|        NPH|              B-Drug|
|    insulin|              I-Drug|
|         in|                   O|
|combination|                   O|
|       with|                   O|
|       oral|                   O|
|     agents|                   O|
|         in|                   O|
|   patients|                   O|
|       with|                   O|
|       type|B-DisorderOrSyndrome|
|          2|I-DisorderOrSyndrome|
|   diabetes|I-DisorderOrSyndrome|
|          .|                   O|
+-----------+--------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_clinical_trials_abstracts|
|Compatibility:|Spark NLP for Healthcare 3.5.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|14.7 MB|

## References

- [Sanchez Graillet O, Cimiano P, Witte C, Ell B. C-TrO: an ontology for summarization and aggregation of the level of evidence in clinical trials. In: Proceedings of the Workshop Ontologies and Data in Life Sciences (ODLS 2019) in the Joint Ontology Workshops' (JOWO 2019). 2019.](https://pub.uni-bielefeld.de/record/2939477)

## Benchmarking

```bash
               label  precision    recall  f1-score   support
               B-Age       0.93      0.81      0.87        16
   B-AllocationRatio       1.00      1.00      1.00         8
            B-Author       0.92      0.97      0.95       335
 B-BioAndMedicalUnit       0.97      0.92      0.94       522
B-CTAnalysisApproach       1.00      0.80      0.89         5
          B-CTDesign       0.95      0.97      0.96       231
        B-Confidence       0.90      0.92      0.91       182
           B-Country       0.95      0.89      0.92       113
B-DisorderOrSyndrome       0.98      0.97      0.98       223
         B-DoseValue       0.94      0.97      0.95       118
              B-Drug       0.97      0.95      0.96      1109
          B-DrugTime       0.95      0.90      0.93       186
          B-Duration       0.88      0.88      0.88       101
           B-Journal       0.96      0.94      0.95        54
    B-NumberPatients       0.94      0.95      0.94       151
              B-PMID       1.00      1.00      1.00        55
            B-PValue       0.87      0.88      0.88       133
B-PercentagePatients       0.92      0.92      0.92       105
   B-PublicationYear       0.86      0.96      0.91        57
         B-TimePoint       0.82      0.79      0.81       231
             B-Value       0.90      0.89      0.90       407
               I-Age       0.83      0.45      0.59        22
   I-AllocationRatio       1.00      1.00      1.00        16
            I-Author       0.93      0.88      0.90       454
 I-BioAndMedicalUnit       0.92      0.97      0.95       263
I-CTAnalysisApproach       1.00      0.89      0.94        18
          I-CTDesign       0.85      0.90      0.88       179
        I-Confidence       0.96      0.95      0.95       717
           I-Country       0.43      0.30      0.35        10
I-DisorderOrSyndrome       1.00      0.98      0.99       345
         I-DoseValue       0.97      0.96      0.96       145
              I-Drug       0.82      0.81      0.82       181
          I-DrugTime       0.99      0.80      0.88       191
          I-Duration       0.89      0.84      0.86       170
           I-Journal       0.94      0.93      0.93       121
    I-NumberPatients       1.00      0.82      0.90        22
            I-PValue       0.96      0.99      0.98       521
I-PercentagePatients       0.93      0.92      0.93       130
         I-TimePoint       0.85      0.69      0.76       283
             I-Value       0.94      0.94      0.94       788
                   O       0.98      0.98      0.98     21613
            accuracy          -         -      0.96     30531
           macro-avg       0.92      0.89      0.90     30531
        weighted-avg       0.96      0.96      0.96     30531
```