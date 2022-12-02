---
layout: model
title: Extract entities in clinical trial abstracts (BertForTokenClassification)
author: John Snow Labs
name: bert_token_classifier_ner_clinical_trials_abstracts
date: 2022-06-29
tags: [berttokenclassifier, bert, biobert, en, ner, licensed]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.5.3
spark_version: 3.0
supported: true
annotator: MedicalBertForTokenClassifier
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This Named Entity Recognition model is trained with the BertForTokenClassification method from transformers library and imported into Spark NLP.

It extracts relevant entities from clinical trial abstracts. It uses a simplified version of the ontology specified by [Sanchez Graillet, O., et al.](https://pub.uni-bielefeld.de/record/2939477) in order to extract concepts related to trial design, diseases, drugs, population, statistics and publication.

## Predicted Entities

`Age`, `AllocationRatio`, `Author`, `BioAndMedicalUnit`, `CTAnalysisApproach`, `CTDesign`, `Confidence`, `Country`, `DisorderOrSyndrome`, `DoseValue`, `Drug`, `DrugTime`, `Duration`, `Journal`, `NumberPatients`, `PMID`, `PValue`, `PercentagePatients`, `PublicationYear`, `TimePoint`, `Value`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_clinical_trials_abstracts_en_3.5.3_3.0_1656475829985.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_clinical_trials_abstracts", "en", "clinical/models")\
.setInputCols("token", "sentence")\
.setOutputCol("ner")\
.setCaseSensitive(True)

ner_converter = NerConverter()\
.setInputCols(["sentence","token","ner"])\
.setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[
documentAssembler,
sentenceDetector,
tokenizer,
tokenClassifier,
ner_converter])

text = ["This open-label, parallel-group, two-arm, pilot study compared the beta-cell protective effect of adding insulin glargine (GLA) vs. NPH insulin to ongoing metformin. Overall, 28 insulin-naive type 2 diabetes subjects (mean +/- SD age, 61.5 +/- 6.7 years; BMI, 30.7 +/- 4.3 kg/m(2)) treated with metformin and sulfonylurea were randomized to add once-daily GLA or NPH at bedtime."]

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

val tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_clinical_trials_abstracts", "en", "clinical/models")
.setInputCols(Array("token", "sentence"))
.setOutputCol("ner")
.setCaseSensitive(True)

val. ner_converter = new NerConverter()
	.setInputCols(Array("sentence","token","ner"))
	.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter))

val text = "This open-label, parallel-group, two-arm, pilot study compared the beta-cell protective effect of adding insulin glargine (GLA) vs. NPH insulin to ongoing metformin. Overall, 28 insulin-naive type 2 diabetes subjects (mean +/- SD age, 61.5 +/- 6.7 years; BMI, 30.7 +/- 4.3 kg/m(2)) treated with metformin and sulfonylurea were randomized to add once-daily GLA or NPH at bedtime."

val data = Seq(text).toDF("text")

val results = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.clinical_trials_abstracts").predict("""This open-label, parallel-group, two-arm, pilot study compared the beta-cell protective effect of adding insulin glargine (GLA) vs. NPH insulin to ongoing metformin. Overall, 28 insulin-naive type 2 diabetes subjects (mean +/- SD age, 61.5 +/- 6.7 years; BMI, 30.7 +/- 4.3 kg/m(2)) treated with metformin and sulfonylurea were randomized to add once-daily GLA or NPH at bedtime.""")
```

</div>

## Results

```bash
+----------------+------------------+
|chunk           |ner_label         |
+----------------+------------------+
|open-label      |CTDesign          |
|parallel-group  |CTDesign          |
|two-arm         |CTDesign          |
|insulin glargine|Drug              |
|GLA             |Drug              |
|NPH insulin     |Drug              |
|metformin       |Drug              |
|28              |NumberPatients    |
|type 2 diabetes |DisorderOrSyndrome|
|61.5            |Age               |
|kg/m(2          |BioAndMedicalUnit |
|metformin       |Drug              |
|sulfonylurea    |Drug              |
|randomized      |CTDesign          |
|once-daily      |DrugTime          |
|GLA             |Drug              |
|NPH             |Drug              |
|bedtime         |DrugTime          |
+----------------+------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_clinical_trials_abstracts|
|Compatibility:|Healthcare NLP 3.5.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|404.3 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- [Sanchez Graillet O, Cimiano P, Witte C, Ell B. C-TrO: an ontology for summarization and aggregation of the level of evidence in clinical trials. In: Proceedings of the Workshop Ontologies and Data in Life Sciences (ODLS 2019) in the Joint Ontology Workshops' (JOWO 2019). 2019.](https://pub.uni-bielefeld.de/record/2939477)

## Benchmarking

```bash
label 				 precision    recall  f1-score   support
B-Age       			0.93      0.88      0.90        16
B-AllocationRatio       1.00      1.00      1.00         7
B-Author       			0.98      1.00      0.99       702
B-BioAndMedicalUnit     0.96      0.97      0.96       723
B-CTAnalysisApproach    1.00      1.00      1.00         5
B-CTDesign       		0.93      0.95      0.94       384
B-Confidence       		0.91      0.95      0.93       184
B-Country       		0.88      0.91      0.90       115
B-DisorderOrSyndrome   	0.92      0.96      0.94       393
B-DoseValue       		0.97      0.98      0.97       117
B-Drug       			0.97      0.98      0.97      3944
B-DrugTime      		0.92      0.90      0.91       202
B-Duration      		0.90      0.88      0.89       100
B-Journal       		1.00      1.00      1.00       131
B-NumberPatients       	0.94      0.98      0.96       165
B-PMID       			1.00      1.00      1.00       239
B-PValue       			0.86      0.89      0.88       132
B-PercentagePatients	0.93      0.97      0.95       105
B-PublicationYear       1.00      0.98      0.99        57
B-TimePoint       		0.78      0.87      0.82       306
B-Value       			0.89      0.87      0.88       407
I-Age       			1.00      0.45      0.62        22
I-AllocationRatio       1.00      1.00      1.00        14
I-Author       			0.99      0.98      0.99       590
I-BioAndMedicalUnit		0.97      0.99      0.98       344
I-CTAnalysisApproach	0.90      1.00      0.95        18
I-CTDesign       		0.84      0.89      0.87       183
I-Confidence       		0.92      0.98      0.95       753
I-Country       		0.00      0.00      0.00        10
I-DisorderOrSyndrome	0.99      0.98      0.99       600
I-DoseValue				0.99      0.98      0.98       164
I-Drug					0.90      0.89      0.90       393
I-DrugTime				0.96      0.80      0.88       192
I-Duration				0.90      0.84      0.87       165
I-Journal       		0.98      0.99      0.99       238
I-NumberPatients        1.00      0.95      0.98        22
I-PValue       			0.96      0.99      0.98       612
I-PercentagePatients	0.99      1.00      1.00       130
I-TimePoint       		0.81      0.78      0.79       282
I-Value       			0.93      0.96      0.95       787
O       				0.99      0.98      0.98     24184
accuracy          		-         -      	0.97     38137
macro-avg       		0.92      0.91      0.91     38137
weighted-avg       		0.97      0.97      0.97     38137
```
