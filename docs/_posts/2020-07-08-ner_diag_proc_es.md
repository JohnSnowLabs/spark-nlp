---
layout: model
title: Detect Diagnosis and Procedures (Spanish)
author: John Snow Labs
name: ner_diag_proc
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
Pretrained named entity recognition deep learning model for diagnostics and procedures in spanish

## Predicted Entities 
``Diagnostico``, ``Procedimiento``

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_diag_proc_es_2.5.3_2.4_1594168623415.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = NerDLModel.pretrained("ner_diag_proc","es","clinical/models")\
	.setInputCols("sentence","token","word_embeddings")\
	.setOutputCol("ner")
```

```scala
val model = NerDLModel.pretrained("ner_diag_proc","es","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|---------------|----------------------------------|
| Name:          | ner_diag_proc                    |
| Type:   | NerDLModel                       |
| Compatibility: | 2.5.3                            |
| License:       | Licensed                         |
| Edition:       | Official                       |
|Input labels:        | [sentence, token, word_embeddings] |
|Output labels:       | [ner]                              |
| Language:      | es                               |
| Dependencies: | embeddings_scielowiki_300d       |

{:.h2_title}
## Data Source
Trained on CodiEsp Challenge dataset trained with `embeddings_scielowiki_300d`
https://temu.bsc.es/codiesp/
