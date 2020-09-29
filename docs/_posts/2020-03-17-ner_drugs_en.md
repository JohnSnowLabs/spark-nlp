---
layout: model
title: Ner DL Model Drugs
author: John Snow Labs
name: ner_drugs
class: NerDLModel
language: en
repository: clinical/models
date: 2020-03-17
tags: [clinical,ner,dl,n2c2,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN.  
Pretrained named entity recognition deep learning model for Drugs.

{:.h2_title}
## Prediction Domain
DrugChem (Drug and Chemicals)

[https://www.i2b2.org/NLP/Medication](https://www.i2b2.org/NLP/Medication)

{:.h2_title}
## Data Source
Trained on i2b2_med7 + FDA with `embeddings_clinical`.

{:.btn-box}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_drugs_en_2.4.4_2.4_1584452534235.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = NerDLModel.pretrained("ner_drugs","en","clinical/models")\
	.setInputCols("sentence","token","word_embeddings")\
	.setOutputCol("ner")
```

```scala
val model = NerDLModel.pretrained("ner_drugs","en","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|----------------|----------------------------------|
| name           | ner_drugs                        |
| model_class    | NerDLModel                       |
| compatibility  | 2.4.4                            |
| license        | Licensed                         |
| edition        | Healthcare                       |
| inputs         | sentence, token, word_embeddings |
| output         | ner                              |
| language       | en                               |
| case_sensitive | False                            |
| upstream_deps  | embeddings_clinical              |

