---
layout: model
title: BGE Reranker V2 M3 Q4_K_M GGUF
author: John Snow Labs
name: bge_reranker_v2_m3_Q4_K_M
date: 2025-09-01
tags: [llamacpp, gguf, reranker, bge, en, open_source]
task: Reranking
language: en
edition: Spark NLP 6.1.2
spark_version: 3.0
supported: true
engine: llamacpp
annotator: AutoGGUFReranker
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Lightweight reranker model, possesses strong multilingual capabilities, easy to deploy, with fast inference.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bge_reranker_v2_m3_Q4_K_M_en_6.1.2_3.0_1756718229635.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bge_reranker_v2_m3_Q4_K_M_en_6.1.2_3.0_1756718229635.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
document = DocumentAssembler() \n    .setInputCol("text") \n    .setOutputCol("document")
reranker = AutoGGUFReranker.pretrained("bge_reranker_v2_m3_Q4_K_M") \n    .setInputCols(["document"]) \n    .setOutputCol("reranked_documents") \n    .setBatchSize(4) \n    .setQuery("A man is eating pasta.")
pipeline = Pipeline().setStages([document, reranker])
data = spark.createDataFrame([
    ["A man is eating food."],
    ["A man is eating a piece of bread."],
    ["The girl is carrying a baby."],
    ["A man is riding a horse."]
]).toDF("text")
result = pipeline.fit(data).transform(data)
result.select("reranked_documents").show(truncate = False)
# Each document will have a relevance_score in metadata showing how relevant it is to the query

```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bge_reranker_v2_m3_Q4_K_M|
|Compatibility:|Spark NLP 6.1.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[reranked_documents]|
|Language:|en|
|Size:|416.0 MB|