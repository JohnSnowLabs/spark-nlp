---
layout: model
title: SentenceResolver ICDO BioBert
author: John Snow Labs
name: biobertresolve_icdo
class: 
language: en
repository: clinical/models
date: 26/10/2020
tags: [clinical,entity_resolver,icdo]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 
Map recognized oncology entities to ICDO codes.

 {:.h2_title}
## Predicted Entities
ICD-O Codes and their normalized definition with BertSentenceEmbeddings: `sent_biobert_pubmed_base_cased` 

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_ICDO/){:.button.button-orange}<br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/biobertresolve_icdo_en_2.6.3_2.4_1603673101579.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = SentenceEntityResolverModel.pretrained("biobertresolve_icdo","en","clinical/models")\
	.setInputCols("sentence_embeddings")\
	.setOutputCol("entity")
```

```scala

```
</div>



{:.model-param}
## Model Information

{:.table-model}
|-------------------------|--------------------------------|
| Model Name              | biobertresolve_icdo            |
| Model Class             | SentenceEntityResolverModel    |
| Spark Compatibility     | 2.6.3                          |
| Spark NLP Compatibility | 2.4                            |
| License                 | Licensed                       |
| Edition                 | Healthcare                     |
| Input Labels            | sentence_embeddings            |
| Output Labels           | entity                         |
| Language                | en                             |
| Case Sensitive          | True                           |
| Upstream Dependencies   | sent_biobert_pubmed_base_cased |




{:.h2_title}
## Data Source
Trained on ICD-O Histology Behaviour dataset

