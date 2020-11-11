---
layout: model
title: ChunkResolver ICD10GM
author: John Snow Labs
name: chunkresolve_ICD10GM
class: ChunkEntityResolverModel
language: de
repository: clinical/models
date: 2020-09-06
tags: [clinical,entity_resolution,de]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Entity Resolution model Based on KNN using Word Embeddings + Word Movers Distance.

## Predicted Entities 
Codes and their normalized definition with `clinical_embeddings`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_ICD10GM_de_2.5.5_2.4_1599431635423.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = ChunkEntityResolverModel.pretrained("chunkresolve_ICD10GM","de","clinical/models")\
	.setInputCols("token","chunk_embeddings")\
	.setOutputCol("entity")
```

```scala
val model = ChunkEntityResolverModel.pretrained("chunkresolve_ICD10GM","de","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("entity")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|----------------|--------------------------|
| Name:           | chunkresolve_ICD10GM     |
| Type:    | ChunkEntityResolverModel |
| Compatibility:  | Spark NLP 2.5.5+                    |
| License:        | Licensed                 |
|Edition:|Official|               |
|Input labels:         | [token, chunk_embeddings]  |
|Output labels:        | [entity]                  |
| Language:       | de                       |
| Case sensitive: | True                     |
| Dependencies:  | w2v_cc_300d              |

{:.h2_title}
## Data Source
FILLUP
