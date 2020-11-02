---
layout: model
title: Recognize Entities DL
author: John Snow Labs
name: recognize_entities_bert
class: PipelineModel
language: en
repository: public/models
date: 2020-08-09
tags: [pipeline]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 




{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/recognize_entities_bert_en_2.0.0_2.4_1599554769343.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = PretrainedPipeline("recognize_entities_bert","en","public/models")

model.annotate("The patient had stomach pain and high fever")
```

```scala
val model = PretrainedPipeline("recognize_entities_bert","en","public/models")

model.annotate("The patient had stomach pain and high fever")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|-------------------------|
| Model Name     | recognize_entities_bert |
| Model Class    | PipelineModel           |
| Dimension      | 2.4                     |
| Compatibility  | 2.0.0                   |
| License        | open source             |
| Edition        | public                  |
| Inputs         |                         |
| Output         |                         |
| Language       | en                      |
| Case Sensitive | True                    |
| Dependencies   |                         |




{:.h2_title}
## Data Source
  
Visit [this]() link to get more information

