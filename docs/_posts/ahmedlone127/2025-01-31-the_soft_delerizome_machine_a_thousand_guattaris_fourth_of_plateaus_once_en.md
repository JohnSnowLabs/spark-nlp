---
layout: model
title: English the_soft_delerizome_machine_a_thousand_guattaris_fourth_of_plateaus_once GPT2Transformer from genaforvena
author: John Snow Labs
name: the_soft_delerizome_machine_a_thousand_guattaris_fourth_of_plateaus_once
date: 2025-01-31
tags: [en, open_source, onnx, text_generation, gpt2]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: onnx
annotator: GPT2Transformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained GPT2Transformer model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`the_soft_delerizome_machine_a_thousand_guattaris_fourth_of_plateaus_once` is a English model originally trained by genaforvena.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/the_soft_delerizome_machine_a_thousand_guattaris_fourth_of_plateaus_once_en_5.5.1_3.0_1738356703034.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/the_soft_delerizome_machine_a_thousand_guattaris_fourth_of_plateaus_once_en_5.5.1_3.0_1738356703034.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")

seq2seq = GPT2Transformer.pretrained("the_soft_delerizome_machine_a_thousand_guattaris_fourth_of_plateaus_once","en") \
      .setInputCols(["documents"]) \
      .setOutputCol("generation")       
        
pipeline = Pipeline().setStages([documentAssembler, seq2seq])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")
    
val seq2seq = GPT2Transformer.pretrained("the_soft_delerizome_machine_a_thousand_guattaris_fourth_of_plateaus_once","en") 
    .setInputCols(Array("documents")) 
    .setOutputCol("generation")

val pipeline = new Pipeline().setStages(Array(documentAssembler, seq2seq))
val data = Seq("I love spark-nlp").toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|the_soft_delerizome_machine_a_thousand_guattaris_fourth_of_plateaus_once|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[generation]|
|Language:|en|
|Size:|467.8 MB|

## References

https://huggingface.co/genaforvena/the_soft_delerizome_machine_a_thousand_guattaris_fourth_of_plateaus_once