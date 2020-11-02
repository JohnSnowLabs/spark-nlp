---
layout: model
title: Biobert Entity Resolver ICDO BioBert
author: John Snow Labs
name: biobertresolve_icdo
class: SentenceEntityResolverModel
language: en
repository: clinical/models
date: 2020-10-26
tags: [clinical,entity_resolver,icdo]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 
Entity Resolution model Based on KNN using Sentence Embeddings, ideally coming from BertSentenceEmbeddings Trained on ICD-O Histology Behaviour dataset

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
val model = SentenceEntityResolverModel.pretrained("biobertresolve_icdo","en","clinical/models")
	.setInputCols("sentence_embeddings")
	.setOutputCol("entity")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|biobertresolve_icdo|
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
Trained on ICD-O Histology Behaviour dataset  
Visit [this](https://apps.who.int/iris/bitstream/handle/10665/96612/9789241548496_eng.pdf) link to get more information

