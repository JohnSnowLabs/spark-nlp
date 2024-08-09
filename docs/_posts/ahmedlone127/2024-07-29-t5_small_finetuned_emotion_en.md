---
layout: model
title: English T5ForConditionalGeneration Small Cased model (from mrm8488)
author: John Snow Labs
name: t5_small_finetuned_emotion
date: 2024-07-29
tags: [en, open_source, t5, onnx]
task: Text Generation
language: en
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

Pretrained T5ForConditionalGeneration model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `t5-small-finetuned-emotion` is a English model originally trained by `mrm8488`.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_small_finetuned_emotion_en_5.4.2_3.0_1722268307293.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_small_finetuned_emotion_en_5.4.2_3.0_1722268307293.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols("text") \
    .setOutputCols("document")

t5 = T5Transformer.pretrained("t5_small_finetuned_emotion","en") \
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
       
val t5 = T5Transformer.pretrained("t5_small_finetuned_emotion","en") 
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
|Model Name:|t5_small_finetuned_emotion|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[output]|
|Language:|en|
|Size:|312.3 MB|

## References

References

- https://huggingface.co/mrm8488/t5-small-finetuned-emotion
- https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html
- https://github.com/dair-ai/emotion_dataset
- https://arxiv.org/pdf/1910.10683.pdf
- https://i.imgur.com/jVFMMWR.png
- https://twitter.com/omarsar0
- https://github.com/dair-ai/emotion_dataset
- https://github.com/patil-suraj/exploring-T5/blob/master/t5_fine_tuning.ipynb
- https://github.com/patil-suraj
- https://i.imgur.com/JBtAwPx.png
- https://twitter.com/mrm8488
- https://www.linkedin.com/in/manuel-romero-cs/