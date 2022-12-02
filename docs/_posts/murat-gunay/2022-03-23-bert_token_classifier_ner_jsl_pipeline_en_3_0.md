---
layout: model
title: Pipeline to Detect Clinical Entities (BertForTokenClassifier)
author: John Snow Labs
name: bert_token_classifier_ner_jsl_pipeline
date: 2022-03-23
tags: [licensed, ner, clinical, bertfortokenclassification, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.4.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


This pretrained pipeline is built on the top of [bert_token_classifier_ner_jsl](https://nlp.johnsnowlabs.com/2022/01/06/bert_token_classifier_ner_jsl_en.html) model.


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_jsl_pipeline_en_3.4.1_3.0_1648044551434.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_ner_jsl_pipeline", "en", "clinical/models")

pipeline.fullAnnotate("The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby-girl also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("bert_token_classifier_ner_jsl_pipeline", "en", "clinical/models")

pipeline.fullAnnotate("The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby-girl also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.")
```
</div>


## Results


```bash
+--------------------------------+------------+
|chunk                           |ner_label   |
+--------------------------------+------------+
|21-day-old                      |Age         |
|Caucasian male                  |Demographics|
|congestion                      |Symptom     |
|mom                             |Demographics|
|yellow discharge                |Symptom     |
|nares                           |Body_Part   |
|she                             |Demographics|
|mild problems with his breathing|Symptom     |
|perioral cyanosis               |Symptom     |
|retractions                     |Symptom     |
|One day ago                     |Date_Time   |
|mom                             |Demographics|
|tactile temperature             |Symptom     |
|Tylenol                         |Drug        |
|Baby-girl                       |Age         |
|decreased p.o. intake           |Symptom     |
|His                             |Demographics|
|breast-feeding                  |Body_Part   |
|his                             |Demographics|
|respiratory congestion          |Symptom     |
+--------------------------------+------------+
only showing top 20 rows
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_jsl_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|404.5 MB|


## Included Models


- DocumentAssembler
- SentenceDetector
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverter
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE4NjIzNDg2NDBdfQ==
-->