---
layout: model
title: Extract entities in clinical trial abstracts
author: John Snow Labs
name: ner_clinical_trials_abstracts
date: 2022-06-22
tags: [ner, clinical, en, licensed]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.5.3
spark_version: 3.0
supported: true
annotator: MedicalNerModel
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
    .setInputCols(["sentence","token", "embeddings"])\
    .setOutputCol("ner")

nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        embeddings,
        clinical_ner])

text = ["A one-year, randomised, multicentre trial comparing insulin glargine with NPH insulin in combination with oral agents in patients with type 2 diabetes. In a multicentre, open, randomised study, 570 patients with Type 2 diabetes, aged 34 - 80 years, were treated for 52 weeks with insulin glargine or NPH insulin given once daily at bedtime."]

data = spark.createDataFrame([text]).toDF("text")

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

val text = "A one-year, randomised, multicentre trial comparing insulin glargine with NPH insulin in combination with oral agents in patients with type 2 diabetes. In a multicentre, open, randomised study, 570 patients with Type 2 diabetes, aged 34 - 80 years, were treated for 52 weeks with insulin glargine or NPH insulin given once daily at bedtime."

val data = Seq(text).toDF("text")

val results = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.ner.clinical_trials_abstracts").predict("""A one-year, randomised, multicentre trial comparing insulin glargine with NPH insulin in combination with oral agents in patients with type 2 diabetes. In a multicentre, open, randomised study, 570 patients with Type 2 diabetes, aged 34 - 80 years, were treated for 52 weeks with insulin glargine or NPH insulin given once daily at bedtime.""")
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
|         In|                   O|
|          a|                   O|
|multicentre|          B-CTDesign|
|          ,|                   O|
|       open|          B-CTDesign|
|          ,|                   O|
| randomised|          B-CTDesign|
|      study|                   O|
|          ,|                   O|
|        570|    B-NumberPatients|
|   patients|                   O|
|       with|                   O|
|       Type|B-DisorderOrSyndrome|
|          2|I-DisorderOrSyndrome|
|   diabetes|I-DisorderOrSyndrome|
|          ,|                   O|
|       aged|                   O|
|         34|               B-Age|
|          -|                   O|
|         80|               B-Age|
|      years|                   O|
|          ,|                   O|
|       were|                   O|
|    treated|                   O|
|        for|                   O|
|         52|          B-Duration|
|      weeks|          I-Duration|
|       with|                   O|
|    insulin|              B-Drug|
|   glargine|              I-Drug|
|         or|                   O|
|        NPH|              B-Drug|
|    insulin|              I-Drug|
|      given|                   O|
|       once|          B-DrugTime|
|      daily|          I-DrugTime|
|         at|                   O|
|    bedtime|          B-DrugTime|
|          .|                   O|
+-----------+--------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_clinical_trials_abstracts|
|Compatibility:|Healthcare NLP 3.5.3+|
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
label           precision    recall  f1-score   support
Age                 0.88      0.61      0.72        38
AllocationRatio     1.00      1.00      1.00        24
Author              0.93      0.92      0.92       789
BioAndMedicalUnit   0.95      0.94      0.95       785
CTAnalysisApproach  1.00      0.87      0.93        23
CTDesign            0.91      0.95      0.93       410
Confidence          0.95      0.95      0.95       899
Country             0.94      0.86      0.90       123
DisorderOrSyndrome  0.99      0.98      0.99       568
DoseValue           0.96      0.97      0.97       263
Drug                0.96      0.95      0.96      1290
DrugTime            0.97      0.85      0.91       377
Duration            0.89      0.86      0.88       271
Journal             0.95      0.93      0.94       175
NumberPatients      0.95      0.94      0.94       173
O                   0.98      0.98      0.98     21613
PMID                1.00      1.00      1.00        55
PValue              0.97      0.99      0.98       654
PercentagePatients  0.92      0.92      0.92       235
PublicationYear     0.86      0.96      0.91        57
TimePoint           0.85      0.75      0.80       514
Value               0.94      0.94      0.94      1195
accuracy            -         -         0.97     30531
macro-avg           0.94      0.91      0.93     30531
weighted-avg        0.97      0.97      0.97     30531
```
