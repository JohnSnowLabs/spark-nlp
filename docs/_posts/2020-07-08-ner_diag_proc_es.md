---
layout: model
title: Ner DL Model Clinical
author: John Snow Labs
name: ner_diag_proc
class: NerDLModel
language: es
repository: clinical/models
date: 2020-07-08
tags: [clinical,ner,codiesp,es]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN.
Pretrained named entity recognition deep learning model for diagnostics and procedures in spanish

{:.h2_title}
## Prediction Domain
Diagnostico, Procedimiento

{:.h2_title}
## Data Source
Trained on CodiEsp Challenge dataset trained with `embeddings_scielowiki_300d`
https://temu.bsc.es/codiesp/

{:.btn-box}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_diag_proc_es_2.5.3_2.4_1594168623415.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = NerDLModel.pretrained("ner_diag_proc","es","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
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
| name          | ner_diag_proc                    |
| model_class   | NerDLModel                       |
| compatibility | 2.5.3                            |
| license       | Licensed                         |
| edition       | Healthcare                       |
| inputs        | sentence, token, word_embeddings |
| output        | ner                              |
| language      | es                               |
| upstream_deps | embeddings_scielowiki_300d       |

