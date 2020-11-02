---
layout: model
title: Entity Recognizer Small
author: John Snow Labs
name: entity_recognizer_sm
class: PipelineModel
language: fi
repository: public/models
date: 2020-01-09
tags: [pipeline]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 




{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_sm_fi_2.6.0_2.4_1598971407192.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = PretrainedPipeline("entity_recognizer_sm","fi","public/models")

model.annotate("The patient had stomach pain and high fever")
```

```scala
val model = PretrainedPipeline("entity_recognizer_sm","fi","public/models")

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
| Compatibility  | 2.6.0                |
| License        | open source          |
| Edition        | public               |
| Inputs         |                      |
| Output         |                      |
| Language       | fi                   |
| Case Sensitive | True                 |
| Dependencies   |                      |




{:.h2_title}
## Data Source
  
Visit [this]() link to get more information

