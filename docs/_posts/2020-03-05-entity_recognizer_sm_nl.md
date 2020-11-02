---
layout: model
title: Entity Recognizer Small
author: John Snow Labs
name: entity_recognizer_sm
class: PipelineModel
language: nl
repository: public/models
date: 2020-03-05
tags: [pipeline]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 




{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_sm_nl_2.5.0_2.4_1588546655907.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = PretrainedPipeline("entity_recognizer_sm","nl","public/models")

model.annotate("The patient had stomach pain and high fever")
```

```scala
val model = PretrainedPipeline("entity_recognizer_sm","nl","public/models")

model.annotate("The patient had stomach pain and high fever")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|----------------------|
| Model Name     | entity_recognizer_sm |
| Model Class    | PipelineModel        |
| Dimension      | 2.4                  |
| Compatibility  | 2.5.0                |
| License        | open source          |
| Edition        | public               |
| Inputs         |                      |
| Output         |                      |
| Language       | nl                   |
| Case Sensitive | True                 |
| Dependencies   |                      |




{:.h2_title}
## Data Source
  
Visit [this]() link to get more information

