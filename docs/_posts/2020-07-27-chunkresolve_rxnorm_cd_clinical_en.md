---
layout: model
title: ChunkResolver Rxnorm Cd Clinical
author: John Snow Labs
name: chunkresolve_rxnorm_cd_clinical
class: ChunkEntityResolverModel
language: en
repository: clinical/models
date: 2020-07-27
tags: [clinical,entity_resolution,rxnorm,cd,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Entity Resolution model Based on KNN using Word Embeddings + Word Movers Distance


{:.h2_title}
## Prediction Domain
RxNorm Codes and their normalized definition with `clinical_embeddings`

{:.h2_title}
## Data Source
Trained on December 2019 RxNorm Clinical Drugs (TTY=CD) ontology graph with `embeddings_clinical`
https://www.nlm.nih.gov/pubs/techbull/nd19/brief/nd19_rxnorm_december_2019_release.html

{:.btn-box}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_rxnorm_cd_clinical_en_2.5.1_2.4_1595813950836.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = ChunkEntityResolverModel.pretrained("chunkresolve_rxnorm_cd_clinical","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("entity")
```

```scala
val model = ChunkEntityResolverModel.pretrained("chunkresolve_rxnorm_cd_clinical","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("entity")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|----------------|---------------------------------|
| name           | chunkresolve_rxnorm_cd_clinical |
| model_class    | ChunkEntityResolverModel        |
| compatibility  | 2.5.1                           |
| license        | Licensed                        |
| edition        | Healthcare                      |
| inputs         | token, chunk_embeddings         |
| output         | entity                          |
| language       | en                              |
| case_sensitive | True                            |
| upstream_deps  | embeddings_clinical             |

