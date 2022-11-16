---
layout: model
title: Adverse Drug Events Classifier (BERT)
author: John Snow Labs
name: bert_sequence_classifier_ade
date: 2022-02-08
tags: [bert, sequence_classification, en, licensed]
task: Text Classification
language: en
edition: Healthcare NLP 3.4.1
spark_version: 3.0
supported: true
annotator: MedicalBertForSequenceClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Classify texts/sentences in two categories:


- `True` : The sentence is talking about a possible ADE.


- `False` : The sentence doesn’t have any information about an ADE.


This model is a [BioBERT-based](https://github.com/dmis-lab/biobert) classifier.


## Predicted Entities


`True`, `False`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_ade_en_3.4.1_3.0_1644324436716.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


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


sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_ade", "en", "clinical/models")\
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
val documenter = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")


val tokenizer = new Tokenizer()
.setInputCols("sentences")
.setOutputCol("token")


val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_ade", "en", "clinical/models")
.setInputCols(Array("document","token"))
.setOutputCol("class")


val pipeline = new Pipeline().setStages(Array(documenter, tokenizer, sequenceClassifier))


val data = Seq("I felt a bit drowsy and had blurred vision after taking Aspirin.").toDF("text")


val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.ade.seq_biobert").predict("""I felt a bit drowsy and had blurred vision after taking Aspirin.""")
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
|Model Name:|bert_sequence_classifier_ade|
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


This model is trained on a custom dataset comprising of CADEC, DRUG-AE and Twimed.


## Benchmarking


```bash 
label  precision  recall  f1-score  support
False       0.97    0.97      0.97     6884
True       0.87    0.85      0.86     1398
accuracy       0.95    0.95      0.95     8282
macro-avg       0.92    0.91      0.91     8282
weighted-avg       0.95    0.95      0.95     8282
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTA5NTEwNTM1MSwxOTk2NjIzMTM3LC0yMD
EzMDEzOTQ1XX0=
-->