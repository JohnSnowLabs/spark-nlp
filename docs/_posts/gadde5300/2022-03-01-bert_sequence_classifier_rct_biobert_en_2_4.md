---
layout: model
title: RCT Classifier (BioBERT)
author: John Snow Labs
name: bert_sequence_classifier_rct_biobert
date: 2022-03-01
tags: [licensed, sequence_classification, bert, en, rct]
task: Text Classification
language: en
edition: Healthcare NLP 3.4.1
spark_version: 2.4
supported: true
annotator: MedicalBertForSequenceClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


This model is a [BioBERT-based](https://github.com/dmis-lab/biobert) classifier that can classify the sections within the abstracts of scientific articles regarding randomized clinical trials (RCT).


## Predicted Entities


`BACKGROUND`, `CONCLUSIONS`, `METHODS`, `OBJECTIVE`, `RESULTS`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_rct_biobert_en_3.4.1_2.4_1646129655723.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_rct_biobert_en_3.4.1_2.4_1646129655723.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")


tokenizer = Tokenizer() \
.setInputCols(["document"]) \
.setOutputCol("token")


sequenceClassifier_loaded = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_rct_biobert", "en", "clinical/models")\
.setInputCols(["document",'token'])\
.setOutputCol("class")


pipeline = Pipeline(stages=[
document_assembler, 
tokenizer,
sequenceClassifier_loaded   
])


data = spark.createDataFrame([["""Previous attempts to prevent all the unwanted postoperative responses to major surgery with an epidural hydrophilic opioid , morphine , have not succeeded . The authors ' hypothesis was that the lipophilic opioid fentanyl , infused epidurally close to the spinal-cord opioid receptors corresponding to the dermatome of the surgical incision , gives equal pain relief but attenuates postoperative hormonal and metabolic responses more effectively than does systemic fentanyl ."""]]).toDF("text")


result = pipeline.fit(data).transform(data)
```
```scala
val documenter = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")


val tokenizer = new Tokenizer()
.setInputCols("document")
.setOutputCol("token")


val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_rct_biobert", "en", "clinical/models")
.setInputCols(Array("document","token"))
.setOutputCol("class")


val pipeline = new Pipeline().setStages(Array(documenter, tokenizer, sequenceClassifier))


val data = Seq("Previous attempts to prevent all the unwanted postoperative responses to major surgery with an epidural hydrophilic opioid , morphine , have not succeeded . The authors ' hypothesis was that the lipophilic opioid fentanyl , infused epidurally close to the spinal-cord opioid receptors corresponding to the dermatome of the surgical incision , gives equal pain relief but attenuates postoperative hormonal and metabolic responses more effectively than does systemic fentanyl .").toDF("text")


val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.clinical_trials").predict("""Previous attempts to prevent all the unwanted postoperative responses to major surgery with an epidural hydrophilic opioid , morphine , have not succeeded . The authors ' hypothesis was that the lipophilic opioid fentanyl , infused epidurally close to the spinal-cord opioid receptors corresponding to the dermatome of the surgical incision , gives equal pain relief but attenuates postoperative hormonal and metabolic responses more effectively than does systemic fentanyl .""")
```

</div>


## Results


```bash
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------+
|text                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |class       |
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------+
|[Previous attempts to prevent all the unwanted postoperative responses to major surgery with an epidural hydrophilic opioid , morphine , have not succeeded . The authors ' hypothesis was that the lipophilic opioid fentanyl , infused epidurally close to the spinal-cord opioid receptors corresponding to the dermatome of the surgical incision , gives equal pain relief but attenuates postoperative hormonal and metabolic responses more effectively than does systemic fentanyl .]|[BACKGROUND]|
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_rct_biobert|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|406.0 MB|
|Case sensitive:|true|
|Max sentence length:|128|


## References


https://arxiv.org/abs/1710.06071


## Benchmarking


```bash
label         precision  recall  f1-score  support
BACKGROUND    0.77       0.86    0.81      2000   
CONCLUSIONS   0.96       0.95    0.95      2000   
METHODS       0.96       0.98    0.97      2000   
OBJECTIVE     0.85       0.77    0.81      2000   
RESULTS       0.98       0.95    0.96      2000   
accuracy      0.9        0.9     0.9       10000  
macro-avg     0.9        0.9     0.9       10000  
weighted-avg  0.9        0.9     0.9       10000  
```
