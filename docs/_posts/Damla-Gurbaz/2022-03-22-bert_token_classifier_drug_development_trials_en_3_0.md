---
layout: model
title: Detect concepts in drug development trials (BertForTokenClassification)
author: John Snow Labs
name: bert_token_classifier_drug_development_trials
date: 2022-03-22
tags: [en, ner, clinical, licensed, bertfortokenclassification]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.3.4
spark_version: 3.0
supported: true
annotator: BertForTokenClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


It is a `BertForTokenClassification` NER model to identify concepts related to drug development including `Trial Groups` , `End Points` , `Hazard Ratio` and other entities in free text.


## Predicted Entities


`Patient_Count`, `Duration`, `End_Point`, `Value`, `Trial_Group`, `Hazard_Ratio`, `Total_Patients`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_DRUGS_DEVELOPMENT_TRIALS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_BERT_TOKEN_CLASSIFIER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_drug_development_trials_en_3.3.4_3.0_1647948437359.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")


sentenceDetector = SentenceDetectorDLModel.pretrained() \
.setInputCols(["document"]) \
.setOutputCol("sentence") 


tokenizer = Tokenizer()\
.setInputCols("sentence")\
.setOutputCol("token")


tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_drug_development_trials", "en", "clinical/models")\
.setInputCols("token", "sentence")\
.setOutputCol("ner")


ner_converter = NerConverter()\
.setInputCols(["sentence","token","ner"])\
.setOutputCol("ner_chunk") 


pipeline =  Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter])     


test_sentence = """In June 2003, the median overall survival  with and without topotecan were 4.0 and 3.6 months, respectively. The best complete response  ( CR ) , partial response  ( PR ) , stable disease and progressive disease were observed in 23, 63, 55 and 33 patients, respectively, with  topotecan,  and 11, 61, 66 and 32 patients, respectively, without topotecan."""


data = spark.createDataFrame([[test_sentence]]).toDF('text')


result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")


val sentenceDetector = SentenceDetectorDLModel.pretrained()
.setInputCols(Array("document"))
.setOutputCol("sentence") 


val tokenizer = new Tokenizer()
.setInputCols(Array("sentence"))
.setOutputCol("token")


val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_drug_development_trials", "en", "clinical/models")
.setInputCols(Array("token", "sentence"))
.setOutputCol("ner")


val ner_converter = NerConverter()
.setInputCols(Array("sentence","token","ner"))
.setOutputCol("ner_chunk")


val pipeline =  new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter))


val data = Seq("In June 2003, the median overall survival  with and without topotecan were 4.0 and 3.6 months, respectively. The best complete response  ( CR ) , partial response  ( PR ) , stable disease and progressive disease were observed in 23, 63, 55 and 33 patients, respectively, with  topotecan,  and 11, 61, 66 and 32 patients, respectively, without topotecan.").toDF("text")


val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.ner.drug_development_trials").predict("""In June 2003, the median overall survival  with and without topotecan were 4.0 and 3.6 months, respectively. The best complete response  ( CR ) , partial response  ( PR ) , stable disease and progressive disease were observed in 23, 63, 55 and 33 patients, respectively, with  topotecan,  and 11, 61, 66 and 32 patients, respectively, without topotecan.""")
```

</div>


## Results


```bash
+-----------------+-------------+
|chunk            |ner_label    |
+-----------------+-------------+
|median           |Duration     |
|overall survival |End_Point    |
|with             |Trial_Group  |
|without topotecan|Trial_Group  |
|4.0              |Value        |
|3.6 months       |Value        |
|23               |Patient_Count|
|63               |Patient_Count|
|55               |Patient_Count|
|33 patients      |Patient_Count|
|topotecan        |Trial_Group  |
|11               |Patient_Count|
|61               |Patient_Count|
|66               |Patient_Count|
|32 patients      |Patient_Count|
|without topotecan|Trial_Group  |
+-----------------+-------------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_drug_development_trials|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|400.9 MB|
|Case sensitive:|true|
|Max sentence length:|256|


## References


Trained on data obtained from `clinicaltrials.gov` and annotated in-house.


## Benchmarking


```bash
label  prec  rec   f1   support
B-Duration  0.93  0.94  0.93    1820
B-End_Point  0.99  0.98  0.98    5022
B-Hazard_Ratio  0.97  0.95  0.96     778
B-Patient_Count  0.81  0.88  0.85     300
B-Trial_Group  0.86  0.88  0.87    6751
B-Value  0.94  0.96  0.95    7675
I-Duration  0.71  0.82  0.76     185
I-End_Point  0.94  0.98  0.96    1491
I-Patient_Count  0.48  0.64  0.55      44
I-Trial_Group  0.78  0.75  0.77    4561
I-Value  0.93  0.95  0.94    1511
O  0.96  0.95  0.95   47423
accuracy  0.94  0.94  0.94   77608
macro-avg  0.79  0.82  0.80   77608
weighted-avg  0.94  0.94  0.94   77608
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE5NTMyODc2OTRdfQ==
-->