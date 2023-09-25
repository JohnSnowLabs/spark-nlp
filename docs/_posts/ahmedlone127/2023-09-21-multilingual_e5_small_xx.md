---
layout: model
title: Multilingual multilingual_e5_small XlmRoBertaSentenceEmbeddings from intfloat
author: John Snow Labs
name: multilingual_e5_small
date: 2023-09-21
tags: [xlm_roberta, xx, open_source, tensorflow]
task: Embeddings
language: xx
edition: Spark NLP 5.1.2
spark_version: 3.0
supported: true
engine: tensorflow
annotator: XlmRoBertaSentenceEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaSentenceEmbeddings  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`multilingual_e5_small` is a Multilingual model originally trained by intfloat.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/multilingual_e5_small_xx_5.1.2_3.0_1695316525385.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/multilingual_e5_small_xx_5.1.2_3.0_1695316525385.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")
    
sentencerDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\ 
    .setInputCols(["document"])\ 
    .setOutputCol("sentence")
    
embeddings =XlmRoBertaSentenceEmbeddings.pretrained("multilingual_e5_small","xx") \
            .setInputCols(["sentence"]) \
            .setOutputCol("embeddings")

pipeline = Pipeline().setStages([document_assembler, sentencerDL, embeddings])

pipelineModel = pipeline.fit(data)

pipelineDF = pipelineModel.transform(data)

```
```scala


val document_assembler = new DocumentAssembler()
    .setInputCol("text") 
    .setOutputCol("documents")
    
val sentencerDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
    .setInputCols(["document"])
    .setOutputCol("sentence")
    
val embeddings = XlmRoBertaSentenceEmbeddings 
    .pretrained("multilingual_e5_small", "xx")
    .setInputCols(Array("sentence")) 
    .setOutputCol("embeddings") 

val pipeline = new Pipeline().setStages(Array(document_assembler, sentencerDL, embeddings))

val pipelineModel = pipeline.fit(data)

val pipelineDF = pipelineModel.transform(data)


```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|multilingual_e5_small|
|Compatibility:|Spark NLP 5.1.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[embeddings]|
|Language:|xx|
|Size:|284.4 MB|

## References

https://huggingface.co/intfloat/multilingual-e5-small