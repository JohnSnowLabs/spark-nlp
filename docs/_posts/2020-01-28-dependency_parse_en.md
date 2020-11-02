---
layout: model
title: Dependency Parse
author: John Snow Labs
name: dependency_parse
class: PipelineModel
language: en
repository: public/models
date: 2020-01-28
tags: [pipeline]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 




{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dependency_parse_en_2.0.2_2.4_1580255669655.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = PretrainedPipeline("dependency_parse","en","public/models")

model.annotate("The patient had stomach pain and high fever")
```

```scala
val model = PretrainedPipeline("dependency_parse","en","public/models")

model.annotate("The patient had stomach pain and high fever")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|------------------|
| Model Name     | dependency_parse |
| Model Class    | PipelineModel    |
| Dimension      | 2.4              |
| Compatibility  | 2.0.2            |
| License        | open source      |
| Edition        | public           |
| Inputs         |                  |
| Output         |                  |
| Language       | en               |
| Case Sensitive | True             |
| Dependencies   |                  |




{:.h2_title}
## Data Source
  
Visit [this]() link to get more information

