---
layout: model
title: Zulu RobertaForMaskedLM Cased model (from MoseliMotsoehli)
author: John Snow Labs
name: roberta_embeddings_zuberta
date: 2022-12-12
tags: [zu, open_source, roberta_embeddings, robertaformaskedlm]
task: Embeddings
language: zu
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForMaskedLM model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `zuBERTa` is a Zulu model originally trained by `MoseliMotsoehli`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_embeddings_zuberta_zu_4.2.4_3.0_1670859948117.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_embeddings_zuberta_zu_4.2.4_3.0_1670859948117.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

roberta_loaded = RoBertaEmbeddings.pretrained("roberta_embeddings_zuberta","zu") \
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
 
val roberta_loaded = RoBertaEmbeddings.pretrained("roberta_embeddings_zuberta","zu") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("embeddings")
    .setCaseSensitive(true)    
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, roberta_loaded))

val data = Seq("I love Spark NLP").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("zu.embed.roberta").predict("""I love Spark NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_embeddings_zuberta|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|zu|
|Size:|256.0 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/MoseliMotsoehli/zuBERTa
- https://wortschatz.uni-leipzig.de/en/download
- https://zu.wikipedia.org/wiki/Special:AllPages