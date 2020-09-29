---
layout: model
title: Explain Clinical Doc Events Relation Assertion
author: John Snow Labs
name: explain_clinical_doc_era
class: PipelineModel
language: en
repository: clinical/models
date: 2020-08-19
tags: [clinical,pipeline,ner,events,assertion,relation,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
  
A pretrained pipeline with ner_clinical_events, assertion_dl and re_temporal_events_clinical. It will extract clinical entities, assign assertion status and find temporal relationships between clinical entities



{:.h2_title}
## Included Models
- ner_clinical_events
- assertion_dl
- re_temporal_events_clinical

{:.btn-box}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/explain_clinical_doc_era_en_2.5.5_2.4_1597845753750.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = PretrainedPipeline("explain_clinical_doc_era","en","clinical/models")

model.annotate("Include a healthcare document here. Can be a prescription, medical note, anything...")
```

```scala
val model = PretrainedPipeline("explain_clinical_doc_era","en","clinical/models")

model.annotate("Include a healthcare document here. Can be a prescription, medical note, anything...")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|---------------|--------------------------|
| name          | explain_clinical_doc_era |
| model_class   | PipelineModel            |
| compatibility | 2.5.5                    |
| license       | Licensed                 |
| edition       | Healthcare               |
| language      | en                       |

