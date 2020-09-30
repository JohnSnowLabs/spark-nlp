---
layout: model
title: ChunkResolver Icdo Clinical
author: John Snow Labs
name: chunkresolve_icdo_clinical
class: ChunkEntityResolverModel
language: en
repository: clinical/models
date: 2020-04-21
tags: [clinical,entity_resolution,icd10,icdo,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Entity Resolution model Based on KNN using Word Embeddings + Word Movers Distance  


{:.h2_title}
## Prediction Labels
ICD-O Codes and their normalized definition with `clinical_embeddings`

[https://apps.who.int/iris/bitstream/handle/10665/96612/9789241548496_eng.pdf](https://apps.who.int/iris/bitstream/handle/10665/96612/9789241548496_eng.pdf)

{:.h2_title}
## Data Source
Trained on ICD-O Histology Behaviour dataset

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_ICDO/){:.button.button-orange}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icdo_clinical_en_2.4.5_2.4_1587491354644.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = ChunkEntityResolverModel.pretrained("chunkresolve_icdo_clinical","en","clinical/models") \
	.setInputCols("token","chunk_embeddings") \
	.setOutputCol("entity")
```

```scala
val model = ChunkEntityResolverModel.pretrained("chunkresolve_icdo_clinical","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("entity")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|----------------|----------------------------|
| Model Name     | chunkresolve_icdo_clinical |
| Type           | ChunkEntityResolverModel   |
| Compatibility  | 2.4.2                      |
| License        | Licensed                   |
| Edition        | Healthcare                 |
| Inputs         | token, chunk_embeddings    |
| Output         | entity                     |
| Language       | en                         |
| Case Sensitive | True                       |
| Dependencies   | embeddings_clinical        |

