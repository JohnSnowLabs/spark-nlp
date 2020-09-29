---
layout: model
title: ChunkResolver Athena Conditions Healthcare
author: John Snow Labs
name: chunkresolve_athena_conditions_healthcare
class: ChunkEntityResolverModel
language: en
repository: clinical/models
date: 2020-09-16
tags: [clinical,entity_resolution,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Entity Resolution model Based on KNN using Word Embeddings + Word Movers Distance


## Predicted Entities 
Athena Codes and their normalized definition

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_athena_conditions_healthcare_en_2.6.0_2.4_1600265258887.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = ChunkEntityResolverModel.pretrained("chunkresolve_athena_conditions_healthcare","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("entity")
```

```scala
val model = ChunkEntityResolverModel.pretrained("chunkresolve_athena_conditions_healthcare","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("entity")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|----------------|-------------------------------------------|
| Name:           | chunkresolve_athena_conditions_healthcare |
| Type:    | ChunkEntityResolverModel                  |
| Compatibility:  | 2.6.0                                     |
| License:        | Licensed                                  |
|Edition:|Official|                                |
|Input labels:         | [token, chunk_embeddings]                   |
|Output labels:        | [entity]                                    |
| Language:       | en                                        |
| Case sensitive: | True                                      |
| Dependencies:  | embeddings_healthcare_100d                |

{:.h2_title}
## Data Source
Trained on Athena dataset