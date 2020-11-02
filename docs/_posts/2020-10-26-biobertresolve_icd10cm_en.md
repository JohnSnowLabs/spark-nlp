---
layout: model
title: SentenceResolver ICD10CM BioBert
author: John Snow Labs
name: biobertresolve_icd10cm
class: 
language: en
repository: clinical/models
date: 26/10/2020
tags: [clinical,entity_resolver,icd10]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 
Map recognized entities to ICD10-CM codes.

 {:.h2_title}
## Predicted Entities
ICD10-CM Codes and their normalized definition with BertSentenceEmbeddings: `sent_biobert_pubmed_base_cased` 

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_ICD10_CM/){:.button.button-orange}<br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/biobertresolve_icd10cm_en_2.6.3_2.4_1603673704767.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = SentenceEntityResolverModel.pretrained("biobertresolve_icd10cm","en","clinical/models")\
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
| Model Name              | biobertresolve_icd10cm         |
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
Trained on ICD10CM Dataset

