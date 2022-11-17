---
layout: model
title: PICO Classifier (BERT)
author: John Snow Labs
name: bert_sequence_classifier_pico_biobert
date: 2022-02-07
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


Classify medical text according to the PICO framework.


This model is a [BioBERT-based](https://github.com/dmis-lab/biobert) classifier.


## Predicted Entities


`CONCLUSIONS`, `DESIGN_SETTING`, `INTERVENTION`, `PARTICIPANTS`, `FINDINGS`, `MEASUREMENTS`, `AIMS`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_pico_biobert_en_3.4.1_3.0_1644265236813.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


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

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_pico", "en", "clinical/models")\
.setInputCols(["document","token"])\
.setOutputCol("class")

pipeline = Pipeline(stages=[
document_assembler, 
tokenizer,
sequenceClassifier    
])

data = spark.createDataFrame([["To compare the results of recording enamel opacities using the TF and modified DDE indices."]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documenter = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer()
.setInputCols(Array("document"))
.setOutputCol("token")

val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_pico_biobert", "en", "clinical/models")
.setInputCols(Array("document","token"))
.setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documenter, tokenizer, sequenceClassifier))

val data = Seq("""To compare the results of recording enamel opacities using the TF and modified DDE indices.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.pico.seq_biobert").predict("""To compare the results of recording enamel opacities using the TF and modified DDE indices.""")
```

</div>


## Results


```bash
+-------------------------------------------------------------------------------------------+------+
|text                                                                                       |result|
+-------------------------------------------------------------------------------------------+------+
|To compare the results of recording enamel opacities using the TF and modified DDE indices.|[AIMS]|
+-------------------------------------------------------------------------------------------+------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_pico_biobert|
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


This model is trained on a custom dataset derived from a PICO classification dataset.


## Benchmarking


```bash
label  precision    recall  f1-score   support
AIMS       0.92      0.94      0.93      3813
CONCLUSIONS       0.85      0.86      0.86      4314
DESIGN_SETTING       0.88      0.78      0.83      5628
FINDINGS       0.91      0.92      0.91      9242
INTERVENTION       0.71      0.78      0.74      2331
MEASUREMENTS       0.79      0.87      0.83      3219
PARTICIPANTS       0.86      0.81      0.83      2723
accuracy         -         -       0.86     31270
macro-avg       0.85      0.85      0.85     31270
weighted-avg       0.87      0.86      0.86     31270
```
