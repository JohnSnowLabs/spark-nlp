---
layout: model
title: starcoder2_3b_int8 model from bigcode
author: John Snow Labs
name: starcoder2_3b_int8
date: 2025-01-30
tags: [en, open_source, openvino]
task: Text Generation
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: openvino
annotator: StarCoderTransformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

StarCoder2-3B model is a 3B parameter model trained on 17 programming languages from The Stack v2, with opt-out requests excluded. The model uses Grouped Query Attention, a context window of 16,384 tokens with a sliding window attention of 4,096 tokens, and was trained using the Fill-in-the-Middle objective on 3+ trillion tokens.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/starcoder2_3b_int8_en_5.5.0_3.0_1738224538904.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/starcoder2_3b_int8_en_5.5.0_3.0_1738224538904.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
data = spark.createDataFrame([
            [1, "def add(a, b):"]]).toDF("id", "text")
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")

starcoder_loaded = StarCoderTransformer \
    .pretrained("starcoder2_3b_int8","en") \
    .setMaxOutputLength(50) \
    .setDoSample(False) \
    .setInputCols(["documents"]) \
    .setOutputCol("generation")

pipeline = Pipeline().setStages([document_assembler, starcoder_loaded])
results = pipeline.fit(data).transform(data)

results.select("generation.result").show(truncate=False)
```
```scala
val documentAssembler = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")
    
val seq2seq = StarCoderTransformer.pretrained("starcoder2_3b_int8","en") 
    .setInputCols(Array("document")) 
    .setOutputCol("generation")

val pipeline = new Pipeline().setStages(Array(documentAssembler, seq2seq))
val data = Seq(""def add(a, b):").toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|starcoder2_3b_int8|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[generation]|
|Language:|en|
|Size:|2.7 GB|

## References

https://huggingface.co/bigcode/starcoder2-3b