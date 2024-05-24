---
layout: model
title: English sent_mxbai_embed_large_v1 BertSentenceEmbeddings from mixedbread-ai
author: John Snow Labs
name: sent_mxbai_embed_large_v1
date: 2024-05-24
tags: [en, open_source, onnx, sentence_embeddings, bert]
task: Embeddings
language: en
edition: Spark NLP 5.2.4
spark_version: 3.0
supported: true
engine: onnx
annotator: BertSentenceEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertSentenceEmbeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_mxbai_embed_large_v1` is a English model originally trained by mixedbread-ai.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_mxbai_embed_large_v1_en_5.2.4_3.0_1716548512298.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_mxbai_embed_large_v1_en_5.2.4_3.0_1716548512298.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")
    
tokenizer = Tokenizer() \ 
      .setInputCols("document") \ 
      .setOutputCol("token")

sentenceDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx") 			.setInputCols(["document"]) 			.setOutputCol("sentence")

embeddings = BertSentenceEmbeddings.pretrained("sent_mxbai_embed_large_v1","en") \
      .setInputCols(["sentence"]) \
      .setOutputCol("embeddings")       
        
pipeline = Pipeline().setStages([documentAssembler, tokenizer, sentenceDL, embeddings])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")
    
val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sentenceDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
	.setInputCols(Array("document"))
	.setOutputCol("sentence")

val embeddings = BertSentenceEmbeddings.pretrained("sent_mxbai_embed_large_v1","en") 
    .setInputCols(Array("sentence")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, sentenceDL, embeddings))
val data = Seq("I love spark-nlp").toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_mxbai_embed_large_v1|
|Compatibility:|Spark NLP 5.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|794.7 MB|

## References

https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1