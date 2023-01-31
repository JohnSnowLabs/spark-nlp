---
layout: model
title: Clean Slang in Texts
author: John Snow Labs
name: clean_slang
date: 2022-06-15
tags: [en, open_source]
task: Text Classification
language: en
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The clean_slang is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps and recognizes entities . It performs most of the common text processing tasks on your dataframe.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/clean_slang_en_4.0.0_3.0_1655323062566.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/clean_slang_en_4.0.0_3.0_1655323062566.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline('clean_slang', lang='en')

testDoc = '''
yo, what is wrong with ya?
'''
```
```scala

val pipeline = new PretrainedPipeline("clean_slang", lang = "en")
val result = pipeline.fullAnnotate("Hello from John Snow Labs ! ")(0)
```

{:.nlu-block}
```python

import nlu
text = [""Hello from John Snow Labs ! ""]
result_df = nlu.load('en.clean.slang').predict(text)
result_df
```
</div>

## Results

```bash

['hey', 'what', 'is', 'wrong', 'with', 'you']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|clean_slang|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|32.1 KB|

## Included Models

- DocumentAssembler
- TokenizerModel
- NormalizerModel