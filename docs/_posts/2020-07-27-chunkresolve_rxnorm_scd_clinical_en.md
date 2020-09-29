---
layout: model
title: ChunkResolver Rxnorm Scd Clinical
author: John Snow Labs
name: chunkresolve_rxnorm_scd_clinical
class: ChunkEntityResolverModel
language: en
repository: clinical/models
date: 2020-07-27
tags: [clinical,entity_resolution,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Entity Resolution model Based on KNN using Word Embeddings + Word Movers Distance

## Predicted Entities 
RxNorm Codes and their normalized definition with `clinical_embeddings`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_rxnorm_scd_clinical_en_2.5.1_2.4_1595813884363.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = ChunkEntityResolverModel.pretrained("chunkresolve_rxnorm_scd_clinical","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("entity")
```

```scala
val model = ChunkEntityResolverModel.pretrained("chunkresolve_rxnorm_scd_clinical","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("entity")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|----------------|----------------------------------|
| Name:           | chunkresolve_rxnorm_scd_clinical |
| Type:    | ChunkEntityResolverModel         |
| Compatibility:  | Spark NLP 2.5.1+                            |
| License:        | Licensed                         |
|Edition:|Official|                       |
|Input labels:         | [token, chunk_embeddings]          |
|Output labels:        | [entity]                           |
| Language:       | en                               |
| Case sensitive: | True                             |
| Dependencies:  | embeddings_clinical              |

{:.h2_title}
## Data Source
Trained on December 2019 RxNorm Clinical Drugs (TTY=SCD) ontology graph with `embeddings_clinical`
https://www.nlm.nih.gov/pubs/techbull/nd19/brief/nd19_rxnorm_december_2019_release.html