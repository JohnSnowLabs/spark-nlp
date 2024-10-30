---
layout: model
title: English gemma_2_2b_it_q4_k_m AutoGGUFModel from lmstudio-community
author: John Snow Labs
name: gemma_2_2b_it_q4_k_m
date: 2024-10-10
tags: [en, open_source, onnx, conversational, text_generation, text_to_text, tensorflow]
task: Text Generation
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: AutoGGUFModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained AutoGGUFModel model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`gemma_2_2b_it_q4_k_m` is a English model prepared by lmstudio-community.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/gemma_2_2b_it_q4_k_m_en_5.5.0_3.0_1728575388230.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/gemma_2_2b_it_q4_k_m_en_5.5.0_3.0_1728575388230.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
             
document = DocumentAssembler() \
       .setInputCol("text") \
       .setOutputCol("document")
    
autoGGUFModel = AutoGGUFModel.pretrained("gemma_2_2b_it_q4_k_m","en") \
    .setInputCols(["document"]) \
    .setOutputCol("completions") \
    .setBatchSize(4) \
    .setNPredict(20) \
    .setNGpuLayers(99) \
    .setTemperature(0.4) \
    .setTopK(40) \
    .setTopP(0.9) \
    .setPenalizeNl(True)

pipeline = Pipeline().setStages([document, autoGGUFModel])
data = spark.createDataFrame([["Hello, I am a"]]).toDF("text")
result = pipeline.fit(data).transform(data)
result.select("completions").show(truncate = False)

```
```scala

val document = new DocumentAssembler()
     .setInputCol("text")
     .setOutputCol("document")
    
val autoGGUFModel = AutoGGUFModel.pretrained("gemma_2_2b_it_q4_k_m", "en")
  .setInputCols("document")
  .setOutputCol("completions")
  .setBatchSize(4)
  .setNPredict(20)
  .setNGpuLayers(99)
  .setTemperature(0.4f)
  .setTopK(40)
  .setTopP(0.9f)
  .setPenalizeNl(true)
                                                                       
val pipeline = new Pipeline().setStages(Array(document, autoGGUFModel))
                                                                       
val data = Seq("Hello, I am a").toDF("text")
val result = pipeline.fit(data).transform(data)
result.select("completions").show(truncate = false)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|gemma_2_2b_it_q4_k_m|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[completions]|
|Language:|en|
|Size:|1.7 GB|

## References

https://huggingface.co/lmstudio-community/gemma-2-2b-it-GGUF