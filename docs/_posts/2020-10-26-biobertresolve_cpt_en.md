---
layout: model
title: Biobert Entity Resolver CPT
author: John Snow Labs
name: biobertresolve_cpt
class: SentenceEntityResolverModel
language: en
repository: clinical/models
date: 2020-10-26
tags: [clinical,entity_resolver,cpt]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 
Entity Resolution model Based on KNN using Sentence Embeddings, ideally coming from BertSentenceEmbeddings Trained on Current Procedural Terminology dataset

 {:.h2_title}
## Predicted Entities
CPT Codes and their normalized definition with BertSentenceEmbeddings: `sent_biobert_pubmed_base_cased` 

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/biobertresolve_cpt_en_2.6.3_2.4_1603673114746.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = SentenceEntityResolverModel.pretrained("biobertresolve_cpt","en","clinical/models")\
	.setInputCols("sentence_embeddings")\
	.setOutputCol("entity")
```

```scala
val model = SentenceEntityResolverModel.pretrained("biobertresolve_cpt","en","clinical/models")
	.setInputCols("sentence_embeddings")
	.setOutputCol("entity")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|biobertresolve_cpt|
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
Trained on Current Procedural Terminology dataset  
Visit [this](https://en.wikipedia.org/wiki/Current_Procedural_Terminology) link to get more information

