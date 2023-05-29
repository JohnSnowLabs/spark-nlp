---
layout: model
title: Clean patterns pipeline for English
author: John Snow Labs
name: clean_pattern
date: 2023-05-27
tags: [open_source, english, clean_pattern, pipeline, en]
task: Named Entity Recognition
language: en
edition: Spark NLP 4.4.2
spark_version: 3.2
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The clean_pattern is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps 
        and recognizes entities .
         It performs most of the common text processing tasks on your dataframe

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/clean_pattern_en_4.4.2_3.2_1685185742070.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/clean_pattern_en_4.4.2_3.2_1685185742070.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('clean_pattern', lang = 'en')
annotations =  pipeline.fullAnnotate(""Hello from John Snow Labs ! "")[0]
annotations.keys()
```
```scala


val pipeline = new PretrainedPipeline("clean_pattern", lang = "en")
val result = pipeline.fullAnnotate("Hello from John Snow Labs ! ")(0)
```

{:.nlu-block}
```python


import nlu
text = [""Hello from John Snow Labs ! ""]
result_df = nlu.load('en.clean.pattern').predict(text)
result_df
```
</div>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('clean_pattern', lang = 'en')
annotations =  pipeline.fullAnnotate(""Hello from John Snow Labs ! "")[0]
annotations.keys()
```
```scala
val pipeline = new PretrainedPipeline("clean_pattern", lang = "en")
val result = pipeline.fullAnnotate("Hello from John Snow Labs ! ")(0)
```

{:.nlu-block}
```python
import nlu
text = [""Hello from John Snow Labs ! ""]
result_df = nlu.load('en.clean.pattern').predict(text)
result_df
```
</div>

## Results

```bash
Results



|    | document   | sentence   | token     | normal    |
|---:|:-----------|:-----------|:----------|:----------|
|  0 | ['Hello']  | ['Hello']  | ['Hello'] | ['Hello'] ||    | document                         | sentence                        | token                                          | normal                                    |


{:.model-param}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|clean_pattern|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|17.2 KB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- NormalizerModel