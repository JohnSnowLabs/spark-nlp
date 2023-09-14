---
layout: model
title: English chemical_bert_uncased_finetuned_cust_c2 BertEmbeddings from shafin
author: John Snow Labs
name: chemical_bert_uncased_finetuned_cust_c2
date: 2023-09-13
tags: [bert, en, open_source, fill_mask, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.1.1
spark_version: 3.0
supported: true
engine: onnx
annotator: BertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertEmbeddings  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`chemical_bert_uncased_finetuned_cust_c2` is a English model originally trained by shafin.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/chemical_bert_uncased_finetuned_cust_c2_en_5.1.1_3.0_1694633093400.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/chemical_bert_uncased_finetuned_cust_c2_en_5.1.1_3.0_1694633093400.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")
    
    
embeddings =BertEmbeddings.pretrained("chemical_bert_uncased_finetuned_cust_c2","en") \
            .setInputCols(["documents","token"]) \
            .setOutputCol("embeddings")

pipeline = Pipeline().setStages([document_assembler, embeddings])

pipelineModel = pipeline.fit(data)

pipelineDF = pipelineModel.transform(data)

```
```scala


val document_assembler = new DocumentAssembler()
    .setInputCol("text") 
    .setOutputCol("embeddings")
    
val embeddings = BertEmbeddings 
    .pretrained("chemical_bert_uncased_finetuned_cust_c2", "en")
    .setInputCols(Array("documents","token")) 
    .setOutputCol("embeddings") 

val pipeline = new Pipeline().setStages(Array(document_assembler, embeddings))

val pipelineModel = pipeline.fit(data)

val pipelineDF = pipelineModel.transform(data)


```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|chemical_bert_uncased_finetuned_cust_c2|
|Compatibility:|Spark NLP 5.1.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|409.1 MB|

## References

https://huggingface.co/shafin/chemical-bert-uncased-finetuned-cust-c2