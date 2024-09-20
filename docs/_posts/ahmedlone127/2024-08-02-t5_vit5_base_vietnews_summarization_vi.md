---
layout: model
title: Vietnamese T5ForConditionalGeneration Base Cased model (from VietAI)
author: John Snow Labs
name: t5_vit5_base_vietnews_summarization
date: 2024-08-02
tags: [vi, open_source, t5, onnx]
task: Text Generation
language: vi
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
engine: onnx
annotator: T5Transformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5ForConditionalGeneration model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `vit5-base-vietnews-summarization` is a Vietnamese model originally trained by `VietAI`.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_vit5_base_vietnews_summarization_vi_5.4.2_3.0_1722631466016.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_vit5_base_vietnews_summarization_vi_5.4.2_3.0_1722631466016.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols("text") \
    .setOutputCols("document")

t5 = T5Transformer.pretrained("t5_vit5_base_vietnews_summarization","vi") \
    .setInputCols("document") \
    .setOutputCol("answers")
    
pipeline = Pipeline(stages=[documentAssembler, t5])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCols("text")
      .setOutputCols("document")
       
val t5 = T5Transformer.pretrained("t5_vit5_base_vietnews_summarization","vi") 
    .setInputCols("document")
    .setOutputCol("answers")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_vit5_base_vietnews_summarization|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[output]|
|Language:|vi|
|Size:|537.3 MB|

## References

References

- https://huggingface.co/VietAI/vit5-base-vietnews-summarization
- https://paperswithcode.com/sota/abstractive-text-summarization-on-vietnews?p=vit5-pretrained-text-to-text-transformer-for
- https://github.com/vietai/ViT5
- https://github.com/vietai/ViT5/blob/main/eval/Eval_vietnews_sum.ipynb