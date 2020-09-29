---
layout: model
title: w2v_cc_300d
author: John Snow Labs
name: ner_healthcare
class: NerDLModel
language: de
repository: clinical/models
date: 2020-09-06
tags: [clinical,ner,events,de]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN.  


{:.h2_title}
## Prediction Domain
BIOLOGICAL_CHEMISTRY,BIOLOGICAL_PARAMETER,BODY_FLUID,BODY_PART,DEGREE,DIAGLAB_PROCEDURE,DOSING,LOCAL_SPECIFICATION,MEASUREMENT,MEDICAL_CONDITION,MEDICAL_DEVICE,MEDICAL_SPECIFICATION,MEDICATION,PERSON,PROCESS,STATE_OF_HEALTH,TIME_INFORMATION,TISSUE,TREATMENT



{:.h2_title}
## Data Source
Trained on 2010 i2b2/VA challenge on concepts, assertions, and relations in clinical text with `w2v_cc_300d`

{:.btn-box}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/14.German_Healthcare_Models.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_healthcare_de_2.5.5_2.4_1599433028253.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = NerDLModel.pretrained("ner_healthcare","de","clinical/models")\
	.setInputCols("sentence","token","word_embeddings")\
	.setOutputCol("ner")
```

```scala
val model = NerDLModel.pretrained("ner_healthcare","de","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|---------------|----------------------------------|
| name          | ner_healthcare                   |
| model_class   | NerDLModel                       |
| compatibility | 2.5.5                            |
| license       | Licensed                         |
| edition       | Healthcare                       |
| inputs        | sentence, token, word_embeddings |
| output        | ner                              |
| language      | de                               |
| upstream_deps | FILLUP                           |

