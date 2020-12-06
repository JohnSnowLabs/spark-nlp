---
layout: model
title: Detect Neoplasms 
author: John Snow Labs
name: ner_neoplasms
class: NerDLModel
language: es
repository: clinical/models
date: 2020-07-08
tags: [clinical,licensed,ner,es]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN.
Neoplasms NER is a Named Entity Recognition model that annotates text to find references to tumors. The only entity it annotates is MalignantNeoplasm. Neoplasms NER is trained with the 'embeddings_scielowiki_300d' word embeddings model, so be sure to use the same embeddings in the pipeline.

## Predicted Entities 
``MORFOLOGIA_NEOPLASIA``

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
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
| Name:           | ner_neoplasms                    |
| Type:    | NerDLModel                       |
| Compatibility:  | Spark NLP 2.5.3+                            |
| License:        | Licensed                         |
|Edition:|Official|                       |
|Input labels:         | [sentence, token, word_embeddings] |
|Output labels:        | [ner]                              |
| Language:       | es                               |
| Case sensitive: | False                            |
| Dependencies:  | embeddings_scielowiki_300d       |

{:.h2_title}
## Data Source
Named Entity Recognition model for Neoplasic Morphology
https://temu.bsc.es/cantemist/