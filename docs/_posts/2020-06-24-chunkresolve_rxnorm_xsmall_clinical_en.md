---
layout: model
title: ChunkResolver Rxnorm Xsmall Clinical
author: John Snow Labs
name: chunkresolve_rxnorm_xsmall_clinical
class: ChunkEntityResolverModel
language: en
repository: clinical/models
date: 2020-06-24
tags: [clinical,entity_resolution,rxnorm,scd,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Entity Resolution model Based on KNN using Word Embeddings + Word Movers Distance  


{:.h2_title}
## Prediction Labels
Snomed Codes and their normalized definition with `clinical_embeddings`

[http://www.snomed.org/](http://www.snomed.org/)

{:.h2_title}
## Data Source
Trained on December 2019 RxNorm Subset

{:.btn-box}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/13.Snomed_Entity_Resolver_Model_Training.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_rxnorm_xsmall_clinical_en_2.5.2_2.4_1592959394598.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = ChunkEntityResolverModel.pretrained("chunkresolve_rxnorm_xsmall_clinical","en","clinical/models") \
	.setInputCols("token","chunk_embeddings") \
	.setOutputCol("entity")
```

```scala
val model = ChunkEntityResolverModel.pretrained("chunkresolve_rxnorm_xsmall_clinical","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("entity")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|----------------|-------------------------------------|
| Model Name     | chunkresolve_rxnorm_xsmall_clinical |
| Type           | ChunkEntityResolverModel            |
| Compatibility  | 2.5.2                               |
| License        | Licensed                            |
| Edition        | Healthcare                          |
| Inputs         | token, chunk_embeddings             |
| Output         | entity                              |
| Language       | en                                  |
| Case Sensitive | True                                |
| Dependencies   | embeddings_clinical                 |

