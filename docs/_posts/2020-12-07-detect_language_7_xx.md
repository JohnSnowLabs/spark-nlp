---
layout: model
title: LanguageDetectorDL
author: John Snow Labs
name: detect_language_7
class: PipelineModel
language: xx
repository: public/models
date: 2020-12-07
tags: [pipeline]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 




{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/detect_language_7_xx_2.5.0_2.4_1594580832687.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = PretrainedPipeline("detect_language_7","xx","public/models")

model.annotate("The patient had stomach pain and high fever")
```

```scala
val model = PretrainedPipeline("detect_language_7","xx","public/models")

model.annotate("The patient had stomach pain and high fever")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|-------------------|
| Model Name     | detect_language_7 |
| Model Class    | PipelineModel     |
| Dimension      | 2.4               |
| Compatibility  | 2.5.0             |
| License        | open source       |
| Edition        | public            |
| Inputs         |                   |
| Output         |                   |
| Language       | xx                |
| Case Sensitive | True              |
| Dependencies   |                   |




{:.h2_title}
## Data Source
  
Visit [this]() link to get more information

