---
layout: model
title: Explain Document ML Pipeline for English
author: John Snow Labs
name: explain_document_ml
date: 2023-05-27
tags: [open_source, english, explain_document_ml, pipeline, en]
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

The explain_document_ml is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps 
and recognizes entities .
It performs most of the common text processing tasks on your dataframe

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_ml_en_4.4.2_3.2_1685186336939.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/explain_document_ml_en_4.4.2_3.2_1685186336939.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


from sparknlp.pretrained import PretrainedPipeline
pipeline = PretrainedPipeline('explain_document_ml', lang = 'en')
annotations =  pipeline.fullAnnotate(""Hello from John Snow Labs ! "")[0]
annotations.keys()
```
```scala


val pipeline = new PretrainedPipeline("explain_document_ml", lang = "en")
val result = pipeline.fullAnnotate("Hello from John Snow Labs ! ")(0)
```

{:.nlu-block}
```python


import nlu
text = [""Hello from John Snow Labs ! ""]
result_df = nlu.load('en.explain').predict(text)
result_df
```
</div>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline
pipeline = PretrainedPipeline('explain_document_ml', lang = 'en')
annotations =  pipeline.fullAnnotate(""Hello from John Snow Labs ! "")[0]
annotations.keys()
```
```scala
val pipeline = new PretrainedPipeline("explain_document_ml", lang = "en")
val result = pipeline.fullAnnotate("Hello from John Snow Labs ! ")(0)
```

{:.nlu-block}
```python
import nlu
text = [""Hello from John Snow Labs ! ""]
result_df = nlu.load('en.explain').predict(text)
result_df
```
</div>

## Results

```bash
Results



|    | document                         | sentence                         | token                                            | spell                                           | lemmas                                          | stems                                          | pos                                    |
|---:|:---------------------------------|:---------------------------------|:-------------------------------------------------|:------------------------------------------------|:------------------------------------------------|:-----------------------------------------------|:---------------------------------------|
|  0 | ['Hello fronm John Snwow Labs!'] | ['Hello fronm John Snwow Labs!'] | ['Hello', 'fronm', 'John', 'Snwow', 'Labs', '!'] | ['Hello', 'front', 'John', 'Snow', 'Labs', '!'] | ['Hello', 'front', 'John', 'Snow', 'Labs', '!'] | ['hello', 'front', 'john', 'snow', 'lab', '!'] | ['UH', 'NN', 'NNP', 'NNP', 'NNP', '.'] ||    | document   | sentence   | token     | spell     | lemmas    | stems     | pos    |


{:.model-param}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|explain_document_ml|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|9.5 MB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- NorvigSweetingModel
- LemmatizerModel
- Stemmer
- PerceptronModel