---
layout: model
title: Check Spelling
author: John Snow Labs
name: check_spelling
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
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/check_spelling_en_2.0.2_2.4_1580246859135.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = PretrainedPipeline("check_spelling","en","public/models")

model.annotate("The patient had stomach pain and high fever")
```

```scala
val model = PretrainedPipeline("check_spelling","en","public/models")

model.annotate("The patient had stomach pain and high fever")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|----------------|
| Model Name     | check_spelling |
| Model Class    | PipelineModel  |
| Dimension      | 2.4            |
| Compatibility  | 2.0.2          |
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

