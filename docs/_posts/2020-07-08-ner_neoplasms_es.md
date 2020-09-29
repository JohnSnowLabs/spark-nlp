---
layout: model
title: Neoplasms NER
author: John Snow Labs
name: ner_neoplasms
class: NerDLModel
language: es
repository: clinical/models
date: 2020-07-08
tags: [clinical,ner,cantemist,es]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN.  
Neoplasms NER is a Named Entity Recognition model that annotates text to find references to tumors. The only entity it annotates is MalignantNeoplasm. Neoplasms NER is trained with the 'embeddings_scielowiki_300d' word embeddings model, so be sure to use the same embeddings in the pipeline.

{:.h2_title}
## Prediction Domain
MORFOLOGIA_NEOPLASIA

[https://temu.bsc.es/cantemist/](https://temu.bsc.es/cantemist/)

{:.h2_title}
## Data Source
Named Entity Recognition model for Neoplasic Morphology

{:.btn-box}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_neoplasms_es_2.5.3_2.4_1594168624415.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = NerDLModel.pretrained("ner_neoplasms","es","clinical/models")\
	.setInputCols("sentence","token","word_embeddings")\
	.setOutputCol("ner")
```

```scala
val model = NerDLModel.pretrained("ner_neoplasms","es","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|----------------|----------------------------------|
| name           | ner_neoplasms                    |
| model_class    | NerDLModel                       |
| compatibility  | 2.5.3                            |
| license        | Licensed                         |
| edition        | Healthcare                       |
| inputs         | sentence, token, word_embeddings |
| output         | ner                              |
| language       | es                               |
| case_sensitive | False                            |
| upstream_deps  | embeddings_scielowiki_300d       |

