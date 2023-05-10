---
layout: model
title: Arabic Electra Embeddings (from aubmindlab)
author: John Snow Labs
name: electra_embeddings_araelectra_base_generator
date: 2022-05-17
tags: [ar, open_source, electra, embeddings]
task: Embeddings
language: ar
edition: Spark NLP 3.4.4
spark_version: 3.0
supported: true
annotator: BertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Electra Embeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `araelectra-base-generator` is a Arabic model orginally trained by `aubmindlab`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/electra_embeddings_araelectra_base_generator_ar_3.4.4_3.0_1652786188141.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/electra_embeddings_araelectra_base_generator_ar_3.4.4_3.0_1652786188141.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
  
embeddings = BertEmbeddings.pretrained("electra_embeddings_araelectra_base_generator","ar") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["أنا أحب الشرارة NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCol("text") 
      .setOutputCol("document")
 
val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("electra_embeddings_araelectra_base_generator","ar") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("أنا أحب الشرارة NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|electra_embeddings_araelectra_base_generator|
|Compatibility:|Spark NLP 3.4.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|ar|
|Size:|222.7 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/aubmindlab/araelectra-base-generator
- https://arxiv.org/pdf/1406.2661.pdf
- https://arxiv.org/abs/2012.15516
- https://archive.org/details/arwiki-20190201
- https://www.semanticscholar.org/paper/1.5-billion-words-Arabic-Corpus-El-Khair/f3eeef4afb81223df96575adadf808fe7fe440b4
- https://www.aclweb.org/anthology/W19-4619
- https://sites.aub.edu.lb/mindlab/
- https://www.yakshof.com/#/
- https://www.behance.net/rahalhabib
- https://www.linkedin.com/in/wissam-antoun-622142b4/
- https://twitter.com/wissam_antoun
- https://github.com/WissamAntoun
- https://www.linkedin.com/in/fadybaly/
- https://twitter.com/fadybaly
- https://github.com/fadybaly