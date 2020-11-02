---
layout: model
title: ADE Pipeline
author: John Snow Labs
name: explain_clinical_doc_ade
class: PipelineModel
language: en
repository: clinical/models
date: 2020-06-10
tags: [clinical,pipeline]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 
A pipeline for Adverse Drug Events (ADE) with ner_ade_biobert, assertiondl_biobert and classifierdl_ade_conversational_biobert. It will extract ADE and DRUG clinical entities, assigen assertion status to ADE entities, and then assign ADE status to a text(True means ADE, False means not related to ADE).Extract adverse drug reaction events and drug entites from text

 {:.h2_title}
## Predicted Entities
ADE, DRUG 

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/explain_clinical_doc_ade_en_2.6.0_2.4_1601455031009.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = PretrainedPipeline("explain_clinical_doc_ade","en","clinical/models")

model.annotate("The patient had stomach pain and high fever")
```

```scala
val model = PretrainedPipeline("explain_clinical_doc_ade","en","clinical/models")

model.annotate("The patient had stomach pain and high fever")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|--------------------------|
| Model Name     | explain_clinical_doc_ade |
| Model Class    | PipelineModel            |
| Dimension      | 2.4                      |
| Compatibility  | 2.6.2                    |
| License        | Licensed                 |
| Edition        | Healthcare               |
| Inputs         |                          |
| Output         |                          |
| Language       | en                       |
| Case Sensitive | True                     |
| Dependencies   |                          |


{:.h2_title}
## Included Models
ner_ade_biobert, assertiondl_biobert, classifierdl_ade_conversational_biobert


{:.h2_title}
## Data Source
  
Visit [this](FILLUP) link to get more information

