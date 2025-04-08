---
layout: model
title: English lsg_bart_base_4096_mediqa_chat_taskb_sectionwise_physical_exam BartTransformer from dhananjay2912
author: John Snow Labs
name: lsg_bart_base_4096_mediqa_chat_taskb_sectionwise_physical_exam
date: 2025-04-07
tags: [en, open_source, onnx, text_generation, bart]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: onnx
annotator: BartTransformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BartTransformer model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`lsg_bart_base_4096_mediqa_chat_taskb_sectionwise_physical_exam` is a English model originally trained by dhananjay2912.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lsg_bart_base_4096_mediqa_chat_taskb_sectionwise_physical_exam_en_5.5.1_3.0_1744066398301.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/lsg_bart_base_4096_mediqa_chat_taskb_sectionwise_physical_exam_en_5.5.1_3.0_1744066398301.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")

seq2seq = BartTransformer.pretrained("lsg_bart_base_4096_mediqa_chat_taskb_sectionwise_physical_exam","en") \
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
    
val seq2seq = BartTransformer.pretrained("lsg_bart_base_4096_mediqa_chat_taskb_sectionwise_physical_exam","en") 
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
|Model Name:|lsg_bart_base_4096_mediqa_chat_taskb_sectionwise_physical_exam|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[output]|
|Language:|en|
|Size:|823.8 MB|

## References

https://huggingface.co/dhananjay2912/lsg-bart-base-4096-mediqa-chat-taskb-sectionwise-physical_exam