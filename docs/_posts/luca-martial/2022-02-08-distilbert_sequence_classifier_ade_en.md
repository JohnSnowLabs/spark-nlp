---
layout: model
title: Adverse Drug Events Classifier (DistilBERT)
author: John Snow Labs
name: distilbert_sequence_classifier_ade
date: 2022-02-08
tags: [bert, sequence_classification, en, licensed]
task: Text Classification
language: en
edition: Healthcare NLP 3.4.1
spark_version: 3.0
supported: true
annotator: MedicalDistilBertForSequenceClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Classify text/sentence in two categories:


`True` : The sentence is talking about a possible ADE


`False` : The sentences doesn’t have any information about an ADE.


This model is a [DistilBERT](https://huggingface.co/distilbert-base-cased)-based classifier. Please note that there is no bio-version of DistilBERT so the performance may not be par with BioBERT-based classifiers.


## Predicted Entities


`True`, `False`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/distilbert_sequence_classifier_ade_en_3.4.1_3.0_1644352732829.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/distilbert_sequence_classifier_ade_en_3.4.1_3.0_1644352732829.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


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


sequenceClassifier = MedicalDistilBertForSequenceClassification.pretrained("distilbert_sequence_classifier_ade", "en", "clinical/models")\
.setInputCols(["document","token"])\
.setOutputCol("class")


pipeline = Pipeline(stages=[
document_assembler, 
tokenizer,
sequenceClassifier    
])


data = spark.createDataFrame([["I felt a bit drowsy and had blurred vision after taking Aspirin."]]).toDF("text")


result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")


val tokenizer = new Tokenizer()
.setInputCols(Array("document"))
.setOutputCol("token")


val sequenceClassifier = MedicalDistilBertForSequenceClassification.pretrained("distilbert_sequence_classifier_ade", "en", "clinical/models")
.setInputCols(Array("document","token"))
.setOutputCol("class")


val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))


val data = Seq("I felt a bit drowsy and had blurred vision after taking Aspirin.").toDF("text")


val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.ade.seq_distilbert").predict("""I felt a bit drowsy and had blurred vision after taking Aspirin.""")
```

</div>


## Results


```bash
+----------------------------------------------------------------+------+
|text                                                            |result|
+----------------------------------------------------------------+------+
|I felt a bit drowsy and had blurred vision after taking Aspirin.|[True]|
+----------------------------------------------------------------+------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|distilbert_sequence_classifier_ade|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|406.3 MB|
|Case sensitive:|true|
|Max sentence length:|128|


## References


This model is trained on a custom dataset comprising of CADEC, DRUG-AE and Twimed.


## Benchmarking


```bash
label  precision  recall  f1-score  support
False       0.93    0.93      0.93     6935
True       0.64    0.65      0.65     1347
accuracy       0.88    0.88      0.88     8282
macro-avg       0.79    0.79      0.79     8282
weighted-avg       0.89    0.88      0.89     8282
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMjA0MjUzNDQ4NywxMDU1NDkzOTk5XX0=
-->