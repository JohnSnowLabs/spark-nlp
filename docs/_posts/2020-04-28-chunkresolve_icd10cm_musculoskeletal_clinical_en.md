---
layout: model
title: ICD10CM Musculoskeletal Entity Resolver
author: John Snow Labs
name: chunkresolve_icd10cm_musculoskeletal_clinical
class: ChunkEntityResolverModel
language: en
repository: clinical/models
date: 2020-04-28
tags: [clinical,entity_resolution,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Entity Resolution model Based on KNN using Word Embeddings + Word Movers Distance

## Predicted Entities 
ICD10-CM Codes and their normalized definition with `clinical_embeddings`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_ICD10_CM/){:.button.button-orange.button-orange-trans.arr.button-icon}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_ICD10_CM.ipynb){:.button.button-orange.button-orange-trans.arr.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icd10cm_musculoskeletal_clinical_en_2.4.5_2.4_1588103998999.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = ChunkEntityResolverModel.pretrained("chunkresolve_icd10cm_musculoskeletal_clinical","en","clinical/models")\
	.setInputCols("token","chunk_embeddings")\
	.setOutputCol("entity")
```

```scala
val model = ChunkEntityResolverModel.pretrained("chunkresolve_icd10cm_musculoskeletal_clinical","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("entity")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|----------------|-----------------------------------------------|
| Name:           | chunkresolve_icd10cm_musculoskeletal_clinical |
| Type:    | ChunkEntityResolverModel                      |
| Compatibility:  | Spark NLP 2.4.5+                                         |
| License:        | Licensed                                      |
|Edition:|Official|                                    |
|Input labels:         | [token, chunk_embeddings]                       |
|Output labels:        | [entity]                                        |
| Language:       | en                                            |
| Case sensitive: | True                                          |
| Dependencies:  | embeddings_clinical                           |

{:.h2_title}
## Data Source
Trained on ICD10CM Dataset Range: M0000-M9979XXS
https://www.icd10data.com/ICD10CM/Codes/M00-M99