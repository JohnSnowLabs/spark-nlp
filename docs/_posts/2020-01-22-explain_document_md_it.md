---
layout: model
title: Explain Document Medium
author: John Snow Labs
name: explain_document_md
class: PipelineModel
language: it
repository: public/models
date: 2020-01-22
tags: [pipeline]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 




{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_md_it_2.0.8_2.4_1579722813892.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = PretrainedPipeline("explain_document_md","it","public/models")

model.annotate("The patient had stomach pain and high fever")
```

```scala
val model = PretrainedPipeline("explain_document_md","it","public/models")

model.annotate("The patient had stomach pain and high fever")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|---------------------|
| Model Name     | explain_document_md |
| Model Class    | PipelineModel       |
| Dimension      | 2.4                 |
| Compatibility  | 2.0.8               |
| License        | open source         |
| Edition        | public              |
| Inputs         |                     |
| Output         |                     |
| Language       | it                  |
| Case Sensitive | True                |
| Dependencies   |                     |




{:.h2_title}
## Data Source
  
Visit [this]() link to get more information

