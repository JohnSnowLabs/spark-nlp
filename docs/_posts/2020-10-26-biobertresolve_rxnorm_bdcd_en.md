---
layout: model
title: SentenceResolver RxNorm Branded and Clinical Drug BioBert
author: John Snow Labs
name: biobertresolve_rxnorm_bdcd
class: 
language: en
repository: clinical/models
date: 26/10/2020
tags: [clinical,entity_resolver,rxnorm]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 
Map recognized drugs to RxNorm codes.

 {:.h2_title}
## Predicted Entities
RxNorm Codes and their normalized definition with BertSentenceEmbeddings: `sent_biobert_pubmed_base_cased` 

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_RXNORM/){:.button.button-orange}<br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/biobertresolve_rxnorm_bdcd_en_2.6.3_2.4_1603683487151.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = SentenceEntityResolverModel.pretrained("biobertresolve_rxnorm_bdcd","en","clinical/models")\
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
| Model Name              | biobertresolve_rxnorm_bdcd     |
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
Trained on RxNorm Clinical Drugs and Branded Drugs

