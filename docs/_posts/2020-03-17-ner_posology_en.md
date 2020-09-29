---
layout: model
title: Ner DL Model Posology
author: John Snow Labs
name: ner_posology
class: NerDLModel
language: en
repository: clinical/models
date: 2020-03-17
tags: [clinical,ner,dl,drugs,fda,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN.
Pretrained named entity recognition deep learning model for posology, this NER is trained with the 'embeddings_clinical' word embeddings model, so be sure to use the same embeddings in the pipeline

{:.h2_title}
## Prediction Domain
DOSAGE,DRUG,DURATION,FORM,FREQUENCY,ROUTE,STRENGTH

{:.h2_title}
## Data Source
Trained on the 2018 i2b2 dataset and FDA Drug datasets with `embeddings_clinical`.
https://open.fda.gov/

{:.btn-box}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_posology_en_2.4.4_2.4_1584452534235.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = NerDLModel.pretrained("ner_posology","en","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
```

```scala
val model = NerDLModel.pretrained("ner_posology","en","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|----------------|----------------------------------|
| name           | ner_posology                     |
| model_class    | NerDLModel                       |
| compatibility  | 2.4.2                            |
| license        | Licensed                         |
| edition        | Healthcare                       |
| inputs         | sentence, token, word_embeddings |
| output         | ner                              |
| language       | en                               |
| case_sensitive | False                            |
| upstream_deps  | embeddings_clinical              |

