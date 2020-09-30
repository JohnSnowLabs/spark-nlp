---
layout: model
title: Ner DL Model Events `embeddings_clinical`
author: John Snow Labs
name: ner_events_clinical
class: NerDLModel
language: en
repository: clinical/models
date: 2020-08-18
tags: [clinical,ner,events,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN.  
Pretrained named entity recognition deep learning model for clinical events.

{:.h2_title}
## Prediction Labels
CLINICAL_DEPT,DATE,DURATION,EVIDENTIAL,FREQUENCY,OCCURRENCE,PROBLEM,TEST,TIME,TREATMENT

[https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/)

{:.h2_title}
## Data Source
Trained on i2b2 events data with `clinical_embeddings`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_EVENTS_CLINICAL/){:.button.button-orange}[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_events_clinical_en_2.5.5_2.4_1597775531760.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = NerDLModel.pretrained("ner_events_clinical","en","clinical/models") \
	.setInputCols("sentence","token","word_embeddings") \
	.setOutputCol("ner")
```

```scala
val model = NerDLModel.pretrained("ner_events_clinical","en","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|----------------|----------------------------------|
| Model Name     | ner_events_clinical              |
| Type           | NerDLModel                       |
| Compatibility  | 2.5.0                            |
| License        | Licensed                         |
| Edition        | Healthcare                       |
| Inputs         | sentence, token, word_embeddings |
| Output         | ner                              |
| Language       | en                               |
| Case Sensitive | False                            |
| Dependencies   | embeddings_clinical              |

