---
layout: model
title: English Bert Embeddings (Cased)
author: John Snow Labs
name: bert_embeddings_lic_class_scancode_bert_base_cased_L32_1
date: 2022-04-11
tags: [bert, embeddings, en, open_source]
task: Embeddings
language: en
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: BertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Bert Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `lic-class-scancode-bert-base-cased-L32-1` is a English model orginally trained by `ayansinha`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_lic_class_scancode_bert_base_cased_L32_1_en_3.4.2_3.0_1649672853388.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_embeddings_lic_class_scancode_bert_base_cased_L32_1_en_3.4.2_3.0_1649672853388.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = BertEmbeddings.pretrained("bert_embeddings_lic_class_scancode_bert_base_cased_L32_1","en") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["I love Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("bert_embeddings_lic_class_scancode_bert_base_cased_L32_1","en") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("I love Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.embed.lic_class_scancode_bert_base_cased_L32_1").predict("""I love Spark NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_lic_class_scancode_bert_base_cased_L32_1|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|en|
|Size:|406.5 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/ayansinha/lic-class-scancode-bert-base-cased-L32-1
- https://github.com/nexB/scancode-results-analyzer
- https://github.com/nexB/scancode-results-analyzer
- https://github.com/nexB/scancode-results-analyzer#quickstart---local-machine
- https://github.com/nexB/scancode-results-analyzer/blob/master/src/results_analyze/nlp_models.py
- https://github.com/nexB/scancode-results-analyzer/blob/master/src/results_analyze/nlp_models.py