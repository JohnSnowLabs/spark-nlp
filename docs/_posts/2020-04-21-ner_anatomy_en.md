---
layout: model
title: Ner DL Model Anatomy
author: John Snow Labs
name: ner_anatomy
class: NerDLModel
language: en
repository: clinical/models
date: 2020-04-21
tags: [clinical,ner,dl,anatomy,anem,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN.
Pretrained named entity recognition deep learning model for anatomy terms.

{:.h2_title}
## Prediction Domain
Anatomical_system,Cell,Cellular_component,Developing_anatomical_structure,Immaterial_anatomical_entity,Organ,Organism_subdivision,Organism_substance,Pathological_formation,Tissue,tissue_structure

{:.h2_title}
## Data Source
Trained on the Anatomical Entity Mention (AnEM) corpus with `embeddings_clinical`
http://www.nactem.ac.uk/anatomy/

{:.btn-box}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_anatomy_en_2.4.2_2.4_1587513307751.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = NerDLModel.pretrained("ner_anatomy","en","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
```

```scala
val model = NerDLModel.pretrained("ner_anatomy","en","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|----------------|----------------------------------|
| name           | ner_anatomy                      |
| model_class    | NerDLModel                       |
| compatibility  | 2.4.2                            |
| license        | Licensed                         |
| edition        | Healthcare                       |
| inputs         | sentence, token, word_embeddings |
| output         | ner                              |
| language       | en                               |
| case_sensitive | False                            |
| upstream_deps  | embeddings_clinical              |

