---
layout: model
title: Gujarati gujarati_xlm_r_base XlmRoBertaEmbeddings from ashwani-tanwar
author: John Snow Labs
name: gujarati_xlm_r_base
date: 2024-08-31
tags: [gu, open_source, onnx, embeddings, xlm_roberta]
task: Embeddings
language: gu
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
engine: onnx
annotator: XlmRoBertaEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaEmbeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`gujarati_xlm_r_base` is a Gujarati model originally trained by ashwani-tanwar.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/gujarati_xlm_r_base_gu_5.4.2_3.0_1725137713751.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/gujarati_xlm_r_base_gu_5.4.2_3.0_1725137713751.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = XlmRoBertaEmbeddings.pretrained("gujarati_xlm_r_base","gu") \
      .setInputCols(["document", "token"]) \
      .setOutputCol("embeddings")       
        
pipeline = Pipeline().setStages([documentAssembler, tokenizer, embeddings])
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

val embeddings = XlmRoBertaEmbeddings.pretrained("gujarati_xlm_r_base","gu") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))
val data = Seq("I love spark-nlp").toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|gujarati_xlm_r_base|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[xlm_roberta]|
|Language:|gu|
|Size:|653.1 MB|

## References

https://huggingface.co/ashwani-tanwar/Gujarati-XLM-R-Base