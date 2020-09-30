---
layout: model
title: Ner DL Model Healthcare
author: John Snow Labs
name: ner_human_phenotype_go_clinical
class: NerDLModel
language: en
repository: clinical/models
date: 2020-08-27
tags: [clinical,ner,phenotype,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN.  


{:.h2_title}
## Prediction Labels
GO,HP



{:.h2_title}
## Data Source


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_HUMAN_PHENOTYPE_GO_CLINICAL/){:.button.button-orange}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_human_phenotype_go_clinical_en_2.5.5_2.4_1598558398770.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = NerDLModel.pretrained("ner_human_phenotype_go_clinical","en","clinical/models") \
	.setInputCols("sentence","token","word_embeddings") \
	.setOutputCol("ner")
```

```scala
val model = NerDLModel.pretrained("ner_human_phenotype_go_clinical","en","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|---------------|----------------------------------|
| Model Name    | ner_human_phenotype_go_clinical  |
| Type          | NerDLModel                       |
| Compatibility | 2.5.5                            |
| License       | Licensed                         |
| Edition       | Healthcare                       |
| Inputs        | sentence, token, word_embeddings |
| Output        | ner                              |
| Language      | en                               |
| Dependencies  | embeddings_clinical              |

