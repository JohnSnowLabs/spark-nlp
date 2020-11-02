---
layout: model
title: Match Datetime
author: John Snow Labs
name: match_datetime
class: PipelineModel
language: en
repository: public/models
date: 2020-10-03
tags: [pipeline]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 




{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/match_datetime_en_2.0.0_2.4_1583845866149.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = PretrainedPipeline("match_datetime","en","public/models")

model.annotate("The patient had stomach pain and high fever")
```

```scala
val model = PretrainedPipeline("match_datetime","en","public/models")

model.annotate("The patient had stomach pain and high fever")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|----------------|
| Model Name     | match_datetime |
| Model Class    | PipelineModel  |
| Dimension      | 2.4            |
| Compatibility  | 2.0.0          |
| License        | open source    |
| Edition        | public         |
| Inputs         |                |
| Output         |                |
| Language       | en             |
| Case Sensitive | True           |
| Dependencies   |                |




{:.h2_title}
## Data Source
  
Visit [this]() link to get more information

