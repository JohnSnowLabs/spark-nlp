---
layout: model
title: English sent_roboust_nlp_xlmr XlmRoBertaSentenceEmbeddings from Blue7Bird
author: John Snow Labs
name: sent_roboust_nlp_xlmr
date: 2024-09-23
tags: [en, open_source, onnx, sentence_embeddings, xlm_roberta]
task: Embeddings
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: XlmRoBertaSentenceEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaSentenceEmbeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_roboust_nlp_xlmr` is a English model originally trained by Blue7Bird.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_roboust_nlp_xlmr_en_5.5.0_3.0_1727062754592.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_roboust_nlp_xlmr_en_5.5.0_3.0_1727062754592.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")

sentenceDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx") \
      .setInputCols(["document"]) \
      .setOutputCol("sentence")

embeddings = XlmRoBertaSentenceEmbeddings.pretrained("sent_roboust_nlp_xlmr","en") \
      .setInputCols(["sentence"]) \
      .setOutputCol("embeddings")       
        
pipeline = Pipeline().setStages([documentAssembler, sentenceDL, embeddings])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")
    
val sentenceDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
	.setInputCols(Array("document"))
	.setOutputCol("sentence")

val embeddings = XlmRoBertaSentenceEmbeddings.pretrained("sent_roboust_nlp_xlmr","en") 
    .setInputCols(Array("sentence")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDL, embeddings))
val data = Seq("I love spark-nlp").toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_roboust_nlp_xlmr|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|1.0 GB|

## References

https://huggingface.co/Blue7Bird/Roboust_nlp_xlmr