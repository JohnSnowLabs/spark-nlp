---
layout: model
title: Detect Clinical Entities (Slim version, BertForTokenClassifier)
author: John Snow Labs
name: bert_token_classifier_ner_jsl_slim
date: 2021-09-24
tags: [ner, bert, en, licensed]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.2.0
spark_version: 2.4
supported: true
annotator: MedicalBertForTokenClassifier
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


This is a pretrained named entity recognition deep learning model for clinical terminology. It is based on the `bert_token_classifier_ner_jsl` model, but with more generalized entities. This model is trained with BertForTokenClassification method from the `transformers` library and imported into Spark NLP.


## Predicted Entities


`Death_Entity`, `Medical_Device`, `Vital_Sign`, `Alergen`, `Drug`, `Clinical_Dept`, `Lifestyle`, `Symptom`, `Body_Part`, `Physical_Measurement`, `Admission_Discharge`, `Date_Time`, `Age`, `Birth_Entity`, `Header`, `Oncological`, `Substance_Quantity`, `Test_Result`, `Test`, `Procedure`, `Treatment`, `Disease_Syndrome_Disorder`, `Pregnancy_Newborn`, `Demographics`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_BERT_TOKEN_CLASSIFIER/){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_BERT_TOKEN_CLASSIFIER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_jsl_slim_en_3.2.0_2.4_1632473007308.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_jsl_slim_en_3.2.0_2.4_1632473007308.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


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
	.setInputCols("sentence")\
	.setOutputCol("token")

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_jsl_slim", "en", "clinical/models")\
	.setInputCols("token", "sentence")\
	.setOutputCol("ner")\
	.setCaseSensitive(True)

ner_converter = NerConverter()\
	.setInputCols(["sentence","token","ner"])\
	.setOutputCol("ner_chunk")

pipeline =  Pipeline(stages=[
	       documentAssembler,
	       sentenceDetector,
	       tokenizer,
	       tokenClassifier,
	       ner_converter])
						       
model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

sample_text = """HISTORY: 30-year-old female presents for digital bilateral mammography secondary to a soft tissue lump palpated by the patient in the upper right shoulder. The patient has a family history of breast cancer within her mother at age 58. Patient denies personal history of breast cancer."""

result = model.transform(spark.createDataFrame([[sample_text]]).toDF("text"))
```
```scala
val documentAssembler = new DocumentAssembler()
	.setInputCol("text")
	.setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")
	.setInputCols("document")
	.setOutputCol("sentence")

val tokenizer = new Tokenizer()
	.setInputCols("sentence")
	.setOutputCol("token")
		
val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_jsl_slim", "en", "clinical/models")
	.setInputCols(Array("token", "sentence"))
	.setOutputCol("ner")
	.setCaseSensitive(True)

val. ner_converter = new NerConverter()
	.setInputCols(Array("sentence","token","ner"))
	.setOutputCol("ner_chunk")

val pipeline =  new Pipeline().setStages(Array(
			documentAssembler,
			sentenceDetector,
			tokenizer,
			tokenClassifier,
			ner_converter))
												
val sample_text = Seq("""HISTORY: 30-year-old female presents for digital bilateral mammography secondary to a soft tissue lump palpated by the patient in the upper right shoulder. The patient has a family history of breast cancer within her mother at age 58. Patient denies personal history of breast cancer.""").toDS.toDF("text")

val result = pipeline.fit(sample_text).transform(sample_text)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.token_bert.ner_jsl_slim").predict("""HISTORY: 30-year-old female presents for digital bilateral mammography secondary to a soft tissue lump palpated by the patient in the upper right shoulder. The patient has a family history of breast cancer within her mother at age 58. Patient denies personal history of breast cancer.""")
```

</div>


## Results


```bash
+----------------+------------+
|chunk           |ner_label   |
+----------------+------------+
|HISTORY:        |Header      |
|30-year-old     |Age         |
|female          |Demographics|
|mammography     |Test        |
|soft tissue lump|Symptom     |
|shoulder        |Body_Part   |
|breast cancer   |Oncological |
|her mother      |Demographics|
|age 58          |Age         |
|breast cancer   |Oncological |
+----------------+------------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_jsl_slim|
|Compatibility:|Healthcare NLP 3.2.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Case sensitive:|true|
|Max sentense length:|256|


## Data Source


Trained on data annotated by JSL.


## Benchmarking


```bash
label  precision    recall  f1-score   support
B-Admission_Discharge       0.82      0.99      0.90       282
B-Age       0.88      0.83      0.85       576
B-Body_Part       0.84      0.91      0.87      8582
B-Clinical_Dept       0.86      0.94      0.90       909
B-Date_Time       0.82      0.77      0.79      1062
B-Death_Entity       0.66      0.98      0.79        43
B-Demographics       0.97      0.98      0.98      5285
B-Disease_Syndrome_Disorder       0.84      0.89      0.86      4259
B-Drug       0.88      0.87      0.87      2555
B-Header       0.97      0.66      0.78      3911
B-Lifestyle       0.77      0.83      0.80       371
B-Medical_Device       0.84      0.87      0.85      3605
B-Oncological       0.86      0.91      0.89       408
B-Physical_Measurement       0.84      0.81      0.82       135
B-Pregnancy_Newborn       0.66      0.71      0.68       245
B-Procedure       0.82      0.88      0.85      2654
B-Symptom       0.83      0.86      0.85      6545
B-Test       0.82      0.83      0.83      2448
B-Test_Result       0.76      0.81      0.78      1280
B-Treatment       0.70      0.76      0.73       275
B-Vital_Sign       0.85      0.87      0.86       627
I-Age       0.84      0.90      0.87       166
I-Alergen       0.00      0.00      0.00         5
I-Body_Part       0.86      0.89      0.88      4946
I-Clinical_Dept       0.92      0.93      0.93       806
I-Date_Time       0.82      0.91      0.86      1173
I-Demographics       0.89      0.84      0.86       416
I-Disease_Syndrome_Disorder       0.87      0.85      0.86      4385
I-Drug       0.83      0.86      0.85      5199
I-Header       0.85      0.97      0.90      6763
I-Lifestyle       0.77      0.69      0.73       134
I-Medical_Device       0.86      0.86      0.86      2341
I-Oncological       0.85      0.94      0.89       515
I-Physical_Measurement       0.88      0.94      0.91       329
I-Pregnancy_Newborn       0.66      0.70      0.68       273
I-Procedure       0.87      0.86      0.87      3414
I-Symptom       0.79      0.75      0.77      6485
I-Test       0.82      0.77      0.79      2283
I-Test_Result       0.67      0.56      0.61       649
I-Treatment       0.69      0.72      0.70       194
I-Vital_Sign       0.88      0.90      0.89       918
O       0.97      0.97      0.97    210520
accuracy        -         -        0.94    297997
macro-avg       0.74      0.74      0.73    297997
weighted-avg       0.94      0.94      0.94    297997
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTE4MjI4MDkzNiwyODQ4MTI0NTYsMTI5Nz
YzNzIzOSwtMjEyMDEzMzQxNywxODg3MzAwOTE3XX0=
-->
