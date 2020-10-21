---
layout: model
title: ChunkResolver Loinc Clinical
author: John Snow Labs
name: chunkresolve_loinc_clinical
class: ChunkEntityResolverModel
language: en
repository: clinical/models
date: 2020-05-16
tags: [clinical,entity_resolution,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Entity Resolution model Based on KNN using Word Embeddings + Word Movers Distance.

## Predicted Entities 
LOINC Codes and ther Standard Name with *clinical_embeddings*.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_loinc_clinical_en_2.5.0_2.4_1589599195201.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = ChunkEntityResolverModel.pretrained("chunkresolve_loinc_clinical","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("entity")
```

```scala
val model = ChunkEntityResolverModel.pretrained("chunkresolve_loinc_clinical","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("entity")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|----------------|-----------------------------|
| Name:           | chunkresolve_loinc_clinical |
| Type:    | ChunkEntityResolverModel    |
| Compatibility:  | Spark NLP 2.5.0+                       |
| License:        | Licensed                    |
|Edition:|Official|                  |
|Input labels:         | [token, chunk_embeddings]     |
|Output labels:        | [entity]                      |
| Language:       | en                          |
| Case sensitive: | True                        |
| Dependencies:  | embeddings_clinical         |

{:.h2_title}
## Data Source
Trained on LOINC dataset with *embeddings_clinical*.
https://loinc.org/
