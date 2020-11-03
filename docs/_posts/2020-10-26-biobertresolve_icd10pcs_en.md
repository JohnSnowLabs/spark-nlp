---
layout: model
title: Biobert Entity Resolver ICD10PCS
author: John Snow Labs
name: biobertresolve_icd10pcs
class: SentenceEntityResolverModel
language: en
repository: clinical/models
date: 2020-10-26
tags: [clinical,entity_resolver,icd10]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 
Entity Resolution model Based on KNN using Sentence Embeddings, ideally coming from BertSentenceEmbeddings Trained on ICD10 Procedure Coding System dataset

 {:.h2_title}
## Predicted Entities
ICD10-PCS Codes and their normalized definition with BertSentenceEmbeddings: `sent_biobert_pubmed_base_cased` 

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/biobertresolve_icd10pcs_en_2.6.3_2.4_1603678407701.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = SentenceEntityResolverModel.pretrained("biobertresolve_icd10pcs","en","clinical/models")\
	.setInputCols("sentence_embeddings")\
	.setOutputCol("entity")
```

```scala
val model = SentenceEntityResolverModel.pretrained("biobertresolve_icd10pcs","en","clinical/models")
	.setInputCols("sentence_embeddings")
	.setOutputCol("entity")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|biobertresolve_icd10pcs|
|Model Class:|SentenceEntityResolverModel|
|Dimension:|2.4|
|Compatibility:|2.6.3|
|License:|Licensed |
|Edition:|Healthcare|
|Inputs:|sentence_embeddings|
|Output:|entity|
|Language:|en|
|Case Sensitive:|True|
|Dependencies:|sent_biobert_pubmed_base_cased|





{:.h2_title}
## Data Source
Trained on ICD10 Procedure Coding System dataset  
Visit [this](https://www.icd10data.com/ICD10PCS/Codes) link to get more information

