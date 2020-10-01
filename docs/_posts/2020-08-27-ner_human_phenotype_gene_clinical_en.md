---
layout: model
title: Ner DL Model Phenotype / Gene
author: John Snow Labs
name: ner_human_phenotype_gene_clinical
class: NerDLModel
language: en
repository: clinical/models
date: 2020-08-27
tags: [clinical,ner,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN.


## Predicted Entities 
GENE, HP

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_human_phenotype_gene_clinical_en_2.5.5_2.4_1598558253840.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = NerDLModel.pretrained("ner_human_phenotype_gene_clinical","en","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
```

```scala
val model = NerDLModel.pretrained("ner_human_phenotype_gene_clinical","en","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---------------|-----------------------------------|
| Name:          | ner_human_phenotype_gene_clinical |
| Type:   | NerDLModel                        |
| Compatibility: | Spark NLP 2.5.5+                             |
| License:       | Licensed                          |
| Edition:       | Official                        |
|Input labels:        | [sentence, token, word_embeddings]  |
|Output labels:       | [ner]                               |
| Language:      | en                                |
| Dependencies: | embeddings_clinical               |

{:.h2_title}
## Data Source