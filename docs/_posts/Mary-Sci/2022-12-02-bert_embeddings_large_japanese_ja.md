---
layout: model
title: Japanese BertForMaskedLM Large Cased model (from cl-tohoku)
author: John Snow Labs
name: bert_embeddings_large_japanese
date: 2022-12-02
tags: [ja, open_source, bert_embeddings, bertformaskedlm]
task: Embeddings
language: ja
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForMaskedLM model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-large-japanese` is a Japanese model originally trained by `cl-tohoku`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_large_japanese_ja_4.2.4_3.0_1670020232480.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_embeddings_large_japanese_ja_4.2.4_3.0_1670020232480.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

bert_loaded = BertEmbeddings.pretrained("bert_embeddings_large_japanese","ja") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings") \
    .setCaseSensitive(True)
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, bert_loaded])

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
 
val bert_loaded = BertEmbeddings.pretrained("bert_embeddings_large_japanese","ja") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("embeddings")
    .setCaseSensitive(True)    
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, bert_loaded))

val data = Seq("I love Spark NLP").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("ja.embed.bert.large").predict("""I love Spark NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_large_japanese|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|ja|
|Size:|1.3 GB|
|Case sensitive:|true|

## References

- https://huggingface.co/cl-tohoku/bert-large-japanese
- https://github.com/google-research/bert
- https://pypi.org/project/unidic-lite/
- https://github.com/cl-tohoku/bert-japanese/tree/v2.0
- https://taku910.github.io/mecab/
- https://github.com/neologd/mecab-ipadic-neologd
- https://github.com/polm/fugashi
- https://github.com/polm/unidic-lite
- https://www.tensorflow.org/tfrc/
- https://creativecommons.org/licenses/by-sa/3.0/
- https://www.tensorflow.org/tfrc/