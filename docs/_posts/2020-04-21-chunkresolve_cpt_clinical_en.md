---
layout: model
title: ChunkResolver Cpt Clinical
author: John Snow Labs
name: chunkresolve_cpt_clinical
class: ChunkEntityResolverModel
language: en
repository: clinical/models
date: 2020-04-21
tags: [clinical,entity_resolution,cpt,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Entity Resolution model Based on KNN using Word Embeddings + Word Movers Distance  


{:.h2_title}
## Prediction Domain
chunkresolve_cpt_clinical Codes and their normalized definition



{:.h2_title}
## Data Source
Trained on Current Procedural Terminology dataset

{:.btn-box}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_cpt_clinical_en_2.4.5_2.4_1587491373378.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = ChunkEntityResolverModel.pretrained("chunkresolve_cpt_clinical","en","clinical/models")\
	.setInputCols("token","chunk_embeddings")\
	.setOutputCol("entity")
```

```scala
val model = ChunkEntityResolverModel.pretrained("chunkresolve_cpt_clinical","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("entity")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|----------------|---------------------------|
| name           | chunkresolve_cpt_clinical |
| model_class    | ChunkEntityResolverModel  |
| compatibility  | 2.4.2                     |
| license        | Licensed                  |
| edition        | Healthcare                |
| inputs         | token, chunk_embeddings   |
| output         | entity                    |
| language       | en                        |
| case_sensitive | True                      |
| upstream_deps  | embeddings_clinical       |

