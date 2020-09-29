---
layout: model
title: Relation Extraction Model Clinical
author: John Snow Labs
name: re_temporal_events_enriched_clinical
class: RelationExtractionModel
language: en
repository: clinical/models
date: 2020-08-18
tags: [clinical,events,relation,extraction,temporal,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Relation Extraction model based on syntactic features using deep learning


{:.h2_title}
## Prediction Domain
Extracts: Temporal relations (BEFORE, AFTER, SIMULTANEOUS, BEGUN_BY, ENDED_BY, DURING, BEFORE_OVERLAP) between clinical events (`ner_events_clinical`)

{:.h2_title}
## Data Source
Trained on data gathered and manually annotated by John Snow Labs
https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

{:.btn-box}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_temporal_events_enriched_clinical_en_2.5.5_2.4_1597775105767.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = RelationExtractionModel.pretrained("re_temporal_events_enriched_clinical","en","clinical/models")
	.setInputCols("word_embeddings","chunk","pos","dependency")
	.setOutputCol("category")
```

```scala
val model = RelationExtractionModel.pretrained("re_temporal_events_enriched_clinical","en","clinical/models")
	.setInputCols("word_embeddings","chunk","pos","dependency")
	.setOutputCol("category")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|----------------|-----------------------------------------|
| name           | re_temporal_events_enriched_clinical    |
| model_class    | RelationExtractionModel                 |
| compatibility  | 2.5.5                                   |
| license        | Licensed                                |
| edition        | Healthcare                              |
| inputs         | word_embeddings, chunk, pos, dependency |
| output         | category                                |
| language       | en                                      |
| case_sensitive | False                                   |
| upstream_deps  | embeddings_clinical                     |

