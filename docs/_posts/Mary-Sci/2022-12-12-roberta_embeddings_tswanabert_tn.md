---
layout: model
title: Tswana RobertaForMaskedLM Cased model (from MoseliMotsoehli)
author: John Snow Labs
name: roberta_embeddings_tswanabert
date: 2022-12-12
tags: [tn, open_source, roberta_embeddings, robertaformaskedlm]
task: Embeddings
language: tn
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForMaskedLM model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `TswanaBert` is a Tswana model originally trained by `MoseliMotsoehli`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_embeddings_tswanabert_tn_4.2.4_3.0_1670858564493.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_embeddings_tswanabert_tn_4.2.4_3.0_1670858564493.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols(["text"]) \
    .setOutputCols("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

roberta_loaded = RoBertaEmbeddings.pretrained("roberta_embeddings_tswanabert","tn") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings") \
    .setCaseSensitive(True)
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, roberta_loaded])

data = spark.createDataFrame([["I love Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
    .setInputCols(Array("text")) 
    .setOutputCols(Array("document"))
      
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
 
val roberta_loaded = RoBertaEmbeddings.pretrained("roberta_embeddings_tswanabert","tn") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("embeddings")
    .setCaseSensitive(true)    
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, roberta_loaded))

val data = Seq("I love Spark NLP").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_embeddings_tswanabert|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|tn|
|Size:|231.0 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/MoseliMotsoehli/TswanaBert
- https://wortschatz.uni-leipzig.de/en/download
- http://doi.org/10.5281/zenodo.3668495
- http://setswana.blogspot.com/
- https://omniglot.com/writing/tswana.php
- http://www.dailynews.gov.bw/
- http://www.mmegi.bw/index.php
- https://tsena.co.bw
- http://www.botswana.co.za/Cultural_Issues-travel/botswana-country-guide-en-route.html
- https://www.poemhunter.com/poem/2013-setswana/
- https://www.poemhunter.com/poem/ngwana-wa-mosetsana/