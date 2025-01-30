---
layout: model
title: English mbert_kmeans_0_05_mmarco_l_all_english_s_all_30k_distill_test_mt5_base_mmarco_v2_30k BertEmbeddings from spear-model
author: John Snow Labs
name: mbert_kmeans_0_05_mmarco_l_all_english_s_all_30k_distill_test_mt5_base_mmarco_v2_30k
date: 2025-01-26
tags: [en, open_source, onnx, embeddings, bert]
task: Embeddings
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: onnx
annotator: BertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertEmbeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mbert_kmeans_0_05_mmarco_l_all_english_s_all_30k_distill_test_mt5_base_mmarco_v2_30k` is a English model originally trained by spear-model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mbert_kmeans_0_05_mmarco_l_all_english_s_all_30k_distill_test_mt5_base_mmarco_v2_30k_en_5.5.1_3.0_1737861537254.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mbert_kmeans_0_05_mmarco_l_all_english_s_all_30k_distill_test_mt5_base_mmarco_v2_30k_en_5.5.1_3.0_1737861537254.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = BertEmbeddings.pretrained("mbert_kmeans_0_05_mmarco_l_all_english_s_all_30k_distill_test_mt5_base_mmarco_v2_30k","en") \
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

val embeddings = BertEmbeddings.pretrained("mbert_kmeans_0_05_mmarco_l_all_english_s_all_30k_distill_test_mt5_base_mmarco_v2_30k","en") 
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
|Model Name:|mbert_kmeans_0_05_mmarco_l_all_english_s_all_30k_distill_test_mt5_base_mmarco_v2_30k|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[bert]|
|Language:|en|
|Size:|665.0 MB|

## References

https://huggingface.co/spear-model/mbert-kmeans-0.05.mmarco.L-all-en.S-all.30K.distill-test.mt5-base-mmarco-v2.30K