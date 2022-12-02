---
layout: model
title: Detect concepts in drug development trials (BertForTokenClassification)
author: John Snow Labs
name: bert_token_classifier_drug_development_trials
date: 2022-06-18
tags: [ner, en, bertfortokenclassification, clinical, licensed]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.4.1
spark_version: 3.0
supported: true
annotator: MedicalBertForTokenClassifier
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

It is a BertForTokenClassification NER model to identify concepts related to drug development including `Trial Groups` , Efficacy and Safety `End Points` , `Hazard Ratio`, and others in free text.

## Predicted Entities

`Hazard_Ratio`, `Confidence_Interval`, `Patient_Count`, `Trial_Group`, `Patient_Group`, `Duration`, `Confidence_level`, `P_Value`, `Confidence_Range`, `End_Point`, `Follow_Up`, `ADE`, `Value`, `DATE`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_DRUGS_DEVELOPMENT_TRIALS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_BERT_TOKEN_CLASSIFIER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_drug_development_trials_en_3.4.1_3.0_1655578771078.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
    .setInputCols("document") 
    .setOutputCol("sentence") 

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
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
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|400.7 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

Trained on data obtained from `clinicaltrials.gov` and annotated in-house.

## Benchmarking

```bash
label                   prec       rec        f1   support
B-ADE                   0.50      0.33      0.40         3
B-Confidence_Interval   0.46      1.00      0.63        12
B-Confidence_Range      1.00      0.98      0.99        42
B-Confidence_level      1.00      0.67      0.81        43
B-DATE                  0.95      0.93      0.94        40
B-Duration              1.00      0.82      0.90        11
B-End_Point             0.91      0.98      0.95        54
B-Follow_Up             1.00      1.00      1.00         2
B-Hazard_Ratio          0.77      1.00      0.87        24
B-P_Value               1.00      0.56      0.71         9
B-Patient_Count         1.00      0.95      0.97        19
B-Patient_Group         0.79      0.63      0.70        43
B-Trial_Group           0.96      0.94      0.95       274
B-Value                 0.98      0.83      0.90        77
I-ADE                   0.71      1.00      0.83        12
I-Confidence_Range      0.98      1.00      0.99        43
I-DATE                  0.95      1.00      0.98        60
I-Duration              1.00      1.00      1.00         1
I-End_Point             0.92      1.00      0.96        44
I-Follow_Up             1.00      1.00      1.00         2
I-P_Value               0.82      1.00      0.90        18
I-Patient_Count         0.00      0.00      0.00         0
I-Patient_Group         0.79      0.94      0.86       187
I-Trial_Group           0.92      0.90      0.91       156
I-Value                 1.00      1.00      1.00        10
O                       0.98      0.98      0.98      2622
accuracy                -         -         0.96      3808
macro-avg               0.86      0.86      0.85      3808
weighted-avg            0.96      0.96      0.96      3808
```