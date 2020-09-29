---
layout: model
title: ChunkResolver Snomed Findings Clinical
author: John Snow Labs
name: chunkresolve_snomed_findings_clinical
class: ChunkEntityResolverModel
language: en
repository: clinical/models
date: 2020-06-20
tags: [clinical,entity_resolution,snomed,findings,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Entity Resolution model Based on KNN using Word Embeddings + Word Movers Distance  


{:.h2_title}
## Prediction Domain
Snomed Codes and their normalized definition with `clinical_embeddings`

[http://www.snomed.org/](http://www.snomed.org/)

{:.h2_title}
## Data Source
Trained on SNOMED CT Findings

{:.btn-box}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/13.Snomed_Entity_Resolver_Model_Training.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_snomed_findings_clinical_en_2.5.1_2.4_1592617161564.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = ChunkEntityResolverModel.pretrained("chunkresolve_snomed_findings_clinical","en","clinical/models")\
	.setInputCols("token","chunk_embeddings")\
	.setOutputCol("entity")
```

```scala
val model = ChunkEntityResolverModel.pretrained("chunkresolve_snomed_findings_clinical","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("entity")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|----------------|---------------------------------------|
| name           | chunkresolve_snomed_findings_clinical |
| model_class    | ChunkEntityResolverModel              |
| compatibility  | 2.5.1                                 |
| license        | Licensed                              |
| edition        | Healthcare                            |
| inputs         | token, chunk_embeddings               |
| output         | entity                                |
| language       | en                                    |
| case_sensitive | True                                  |
| upstream_deps  | embeddings_clinical                   |

