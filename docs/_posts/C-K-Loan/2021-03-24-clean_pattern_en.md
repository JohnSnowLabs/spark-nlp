---
layout: model
title: Clean patterns pipeline for English
author: John Snow Labs
name: clean_pattern
date: 2021-03-24
tags: [open_source, english, clean_pattern, pipeline, en]
supported: true
task: [Named Entity Recognition, Lemmatization]
language: en
edition: Spark NLP 3.0.0
spark_version: 3.0
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The clean_pattern is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps 
        and recognizes entities .
         It performs most of the common text processing tasks on your dataframe

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN_18/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/clean_pattern_en_3.0.0_3.0_1616544446008.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/clean_pattern_en_3.0.0_3.0_1616544446008.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

## Results

```bash
|    | document   | sentence   | token     | normal    |
|---:|:-----------|:-----------|:----------|:----------|
|  0 | ['Hello']  | ['Hello']  | ['Hello'] | ['Hello'] ||    | document                         | sentence                        | token                                          | normal                                    |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|clean_pattern|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|