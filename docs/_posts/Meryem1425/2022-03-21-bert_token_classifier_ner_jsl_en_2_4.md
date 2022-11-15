---
layout: model
title: Detect Clinical Entities (BertForTokenClassifier)
author: John Snow Labs
name: bert_token_classifier_ner_jsl
date: 2022-03-21
tags: [ner_jsl, ner, berfortokenclassification, en, licensed]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.3.4
spark_version: 2.4
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---




## Description




Pretrained named entity recognition deep learning model for clinical terminology. This model is trained with `BertForTokenClassification` method from `transformers` library and imported into Spark NLP. It detects 77 entities.




## Predicted Entities

`Medical_Device`, `Physical_Measurement`, `Alergen`, `Procedure`, `Substance_Quantity`, `Drug`, `Test_Result`, `Pregnancy_Newborn`, `Admission_Discharge`, `Demographics`, `Lifestyle`, `Header`, `Date_Time`, `Treatment`, `Clinical_Dept`, `Test`, `Death_Entity`, `Age`, `Oncological`, `Body_Part`, `Birth_Entity`, `Vital_Sign`, `Symptom`, `Disease_Syndrome_Disorder`




{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_BERT_TOKEN_CLASSIFIER/){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_BERT_TOKEN_CLASSIFIER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_jsl_en_3.3.4_2.4_1647895738040.zip){:.button.button-orange.button-orange-trans.arr.button-icon}




## How to use


<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
	.setInputCol("text")\
	.setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
  .setInputCols(["document"])\
  .setOutputCol("sentence")

tokenizer = Tokenizer()\
  .setInputCols("sentence")\
  .setOutputCol("token")

tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_jsl", "en", "clinical/models")\
  .setInputCols(["token", "sentence"])\
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
						       

sample_text = """The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby-girl also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature."""

df = spark.createDataFrame([[sample_text]]).toDF("text")

result = pipeline.fit(df).transform(df)
```
```scala
val documentAssembler = new DocumentAssembler()
	.setInputCol("text")
	.setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")
	.setInputCols(Array("document"))
	.setOutputCol("sentence")

val tokenizer = new Tokenizer()
	.setInputCols("sentence")
	.setOutputCol("token")
		
val tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_jsl", "en", "clinical/models")
.setInputCols(Array("token", "sentence"))
.setOutputCol("ner")
.setCaseSensitive(True)

val ner_converter = new NerConverter()
	.setInputCols(Array("sentence","token","ner"))
	.setOutputCol("ner_chunk")

val pipeline =  new Pipeline().setStages(Array(
		documentAssembler,
		sentenceDetector,
		tokenizer,
		tokenClassifier,
		ner_converter))
												
val sample_text = Seq("""The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby-girl also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.""").toDS.toDF("text")

val result = pipeline.fit(sample_text).transform(sample_text)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.token_bert.ner_jsl").predict("""The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby-girl also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.""")
```

</div>




## Results


```bash
+--------------------------------+-------------+
|chunk                           |ner_label    |
+--------------------------------+-------------+
|21-day-old                      |Age          |
|Caucasian male                  |Demographics |
|congestion                      |Symptom      |
|mom                             |Demographics |
|yellow discharge                |Symptom      |
|nares                           |Body_Part    |
|she                             |Demographics |
|mild problems with his breathing|Symptom      |
|perioral cyanosis               |Symptom      |
|retractions                     |Symptom      |
|One day ago                     |Date_Time    |
|mom                             |Demographics |
|tactile temperature             |Symptom      |
|Tylenol                         |Drug         |
|Baby-girl                       |Age          |
|decreased p.o. intake           |Symptom      |
|His                             |Demographics |
|breast-feeding                  |Body_Part    |
|his                             |Demographics |
|respiratory congestion          |Symptom      |
|He                              |Demographics |
|tired                           |Symptom      |
|fussy                           |Symptom      |
|over the past 2 days            |Date_Time    |
|albuterol                       |Drug         |
|ER                              |Clinical_Dept|
|His                             |Demographics |
|urine output has                |Symptom      |
|decreased                       |Symptom      |
|he                              |Demographics |
|he                              |Demographics |
|Mom                             |Demographics |
|diarrhea                        |Symptom      |
|His                             |Demographics |
|bowel                           |Body_Part    |
+--------------------------------+-------------+
```




{:.model-param}
## Model Information




{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_jsl|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|404.5 MB|
|Case sensitive:|true|
|Max sentence length:|256|




## Benchmarking

## Benchmarking


```bash
label                        tp     fp     fn     prec       rec        f1       
B-Medical_Device             2696   444    282    0.8585987  0.9053055  0.8813337
I-Physical_Measurement       220    16     34     0.9322034  0.8661417  0.8979592
B-Procedure                  1800   239    281    0.8827857  0.8649688  0.8737864
B-Drug                       1865   218    237    0.8953432  0.8872502  0.8912784
I-Test_Result                289    203    292    0.5873983  0.4974182  0.5386766
I-Pregnancy_Newborn          150    41     104    0.7853403  0.5905512  0.6741573
B-Admission_Discharge        255    35     6      0.8793103  0.9770115  0.9255898
B-Demographics               4609   119    123    0.9748308  0.9740068  0.9744186
I-Lifestyle                  71     49     20     0.5916666  0.7802198  0.6729857
B-Header                     2463   53     122    0.9789348  0.9528046  0.965693 
I-Date_Time                  928    184    191    0.8345324  0.8293119  0.8319139
B-Test_Result                866    198    262    0.8139097  0.7677305  0.7901459
I-Treatment                  114    37     46     0.7549669  0.7125     0.733119 
B-Clinical_Dept              688    83     76     0.8923476  0.9005235  0.8964169
B-Test                       1920   333    313    0.8521970  0.8598298  0.8559965
B-Death_Entity               36     9      2      0.8        0.9473684  0.8674699
B-Lifestyle                  268    58     50     0.8220859  0.8427673  0.8322981
B-Date_Time                  823    154    176    0.8423746  0.8238238  0.8329959
I-Age                        136    34     49     0.8        0.7351351  0.7661972
I-Oncological                345    41     19     0.8937824  0.9478022  0.9199999
I-Body_Part                  3717   720    424    0.8377282  0.8976093  0.8666356
B-Pregnancy_Newborn          153    51     104    0.75       0.5953307  0.6637744
B-Treatment                  169    41     58     0.8047619  0.7444933  0.7734553
I-Procedure                  2302   326    417    0.8759513  0.8466348  0.8610435
B-Birth_Entity               6      5      7      0.5454545  0.4615384  0.5      
I-Vital_Sign                 639    197    93     0.7643540  0.8729508  0.815051 
I-Header                     4451   111    216    0.9756685  0.9537176  0.9645682
I-Death_Entity               2      0      0      1          1          1        
I-Clinical_Dept              621    54     39     0.92       0.9409091  0.9303371
I-Test                       1593   378    353    0.8082192  0.8186022  0.8133775
B-Age                        472    43     51     0.9165048  0.9024856  0.9094413
I-Symptom                    4227   1271   1303   0.7688250  0.7643761  0.7665941
I-Demographics               321    53     53     0.8582887  0.8582887  0.8582887
B-Body_Part                  6312   912    809    0.8737541  0.8863923  0.8800279
B-Physical_Measurement       91     10     17     0.9009901  0.8425926  0.8708134
B-Disease_Syndrome_Disorder  2817   336    433    0.8934348  0.8667692  0.8799001
B-Symptom                    4522   830    747    0.8449178  0.8582274  0.8515206
I-Disease_Syndrome_Disorder  2814   386    530    0.879375   0.8415072  0.8600244
I-Drug                       3737   612    517    0.859278   0.8784673  0.8687667
I-Medical_Device             1825   331    131    0.8464749  0.9330266  0.8876459
B-Oncological                276    28     27     0.9078947  0.9108911  0.9093904
B-Vital_Sign                 429    97     79     0.8155893  0.8444882  0.8297872
Macro-average                62038  9340   9110   0.7678277  0.7648211  0.7663215
Micro-average                62038  9340   9110   0.8691473  0.8719570  0.87055  
```

