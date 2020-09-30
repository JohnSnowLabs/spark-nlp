---
layout: model
title: Ner DL Model Clinical (Large)
author: John Snow Labs
name: ner_large_clinical
class: NerDLModel
language: en
repository: clinical/models
date: 2020-05-21
tags: [clinical,ner,generic,large,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN.  


{:.h2_title}
## Prediction Labels
PROBLEM,TEST,TREATMENT

[https://www.johnsnowlabs.com/data/](https://www.johnsnowlabs.com/data/)

{:.h2_title}
## Data Source
Trained on data gathered and manually annotated by John Snow Labs

{:.btn-box}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_large_clinical_en_2.5.0_2.4_1590021302624.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = NerDLModel.pretrained("ner_large_clinical","en","clinical/models") \
	.setInputCols("sentence","token","word_embeddings") \
	.setOutputCol("ner")
```

```scala
val model = NerDLModel.pretrained("ner_large_clinical","en","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|---------------|----------------------------------|
| Model Name    | ner_large_clinical               |
| Type          | NerDLModel                       |
| Compatibility | 2.5.0                            |
| License       | Licensed                         |
| Edition       | Healthcare                       |
| Inputs        | sentence, token, word_embeddings |
| Output        | ner                              |
| Language      | en                               |
| Dependencies  | embeddings_clinical              |

