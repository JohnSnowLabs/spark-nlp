---
layout: model
title: Match Chunks in Texts
author: John Snow Labs
name: match_chunks
date: 2023-05-21
tags: [en, open_source]
task: Text Classification
language: en
edition: Spark NLP 4.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The pipeline uses regex `<DT/>?/<JJ/>*<NN>+`

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/match_chunks_en_4.4.2_3.0_1684628637228.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/match_chunks_en_4.4.2_3.0_1684628637228.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

from sparknlp.pretrained import PretrainedPipeline

pipeline_local = PretrainedPipeline('match_chunks')

result = pipeline_local.annotate("David visited the restaurant yesterday with his family. He also visited and the day before, but at that time he was alone. David again visited today with his colleagues. He and his friends really liked the food and hoped to visit again tomorrow.")

result['chunk']
```
```scala

import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline import com.johnsnowlabs.nlp.SparkNLP

SparkNLP.version()

val testData = spark.createDataFrame(Seq( (1, "David visited the restaurant yesterday with his family. He also visited and the day before, but at that time he was alone. David again visited today with his colleagues. He and his friends really liked the food and hoped to visit again tomorrow."))).toDF("id", "text")

val pipeline = PretrainedPipeline("match_chunks", lang="en")

val annotation = pipeline.transform(testData)

annotation.show()
```


{:.nlu-block}
```python
import nlu
nlu.load("en.match.chunks").predict("""David visited the restaurant yesterday with his family. He also visited and the day before, but at that time he was alone. David again visited today with his colleagues. He and his friends really liked the food and hoped to visit again tomorrow.""")
```

</div>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline_local = PretrainedPipeline('match_chunks')

result = pipeline_local.annotate("David visited the restaurant yesterday with his family. He also visited and the day before, but at that time he was alone. David again visited today with his colleagues. He and his friends really liked the food and hoped to visit again tomorrow.")

result['chunk']
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline import com.johnsnowlabs.nlp.SparkNLP

SparkNLP.version()

val testData = spark.createDataFrame(Seq( (1, "David visited the restaurant yesterday with his family. He also visited and the day before, but at that time he was alone. David again visited today with his colleagues. He and his friends really liked the food and hoped to visit again tomorrow."))).toDF("id", "text")

val pipeline = PretrainedPipeline("match_chunks", lang="en")

val annotation = pipeline.transform(testData)

annotation.show()
```

{:.nlu-block}
```python
import nlu
nlu.load("en.match.chunks").predict("""David visited the restaurant yesterday with his family. He also visited and the day before, but at that time he was alone. David again visited today with his colleagues. He and his friends really liked the food and hoped to visit again tomorrow.""")
```
</div>

## Results

```bash
Results



['the restaurant yesterday',
'family',
'the day',
'that time',
'today',
'the food',
'tomorrow']


{:.model-param}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|match_chunks|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|4.1 MB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- PerceptronModel
- Chunker