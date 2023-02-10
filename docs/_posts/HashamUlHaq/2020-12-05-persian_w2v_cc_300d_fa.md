---
layout: model
title: Word Embeddings for Persian (persian_w2v_cc_300d)
author: John Snow Labs
name: persian_w2v_cc_300d
date: 2020-12-05
task: Embeddings
language: fa
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [embeddings, fa, open_source]
supported: true
annotator: WordEmbeddingsModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is trained on Common Crawl and Wikipedia using fastText. It is trained using CBOW with position-weights, in dimension 300, with character n-grams of length 5, a window of size 5 and 10 negatives.

The model gives 300 dimensional vector outputs per token. The output vectors map words into a meaningful space where the distance between the vectors is related to semantic similarity of words.

These embeddings can be used in multiple tasks like semantic word similarity, named entity recognition, sentiment analysis, and classification.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/persian_w2v_cc_300d_fa_2.7.0_2.4_1607169840793.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/persian_w2v_cc_300d_fa_2.7.0_2.4_1607169840793.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

Use as part of a pipeline after tokenization.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
embeddings = WordEmbeddingsModel.pretrained("persian_w2v_cc_300d", "fa") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
result = pipeline_model.transform(spark.createDataFrame([['من یادگیری ماشین را دوست دارم']], ["text"]))
```

```scala
val embeddings = WordEmbeddingsModel.pretrained("persian_w2v_cc_300d", "fa") 
.setInputCols(Array("document", "token"))
.setOutputCol("embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
val data = Seq("من یادگیری ماشین را دوست دارم").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""من یادگیری ماشین را دوست دارم"""]
farvec_df = nlu.load('fa.embed.word2vec.300d').predict(text, output_level='token')
farvec_df
```

</div>

{:.h2_title}
## Results
The model gives 300 dimensional Word2Vec feature vector outputs per token.
```bash
| token	| fa_embed_word2vec_300d_embeddings
|-------|--------------------------------------------------		
| من	| [-0.3861289620399475, -0.08295578509569168, -0...
| را	| [-0.15430298447608948, -0.24924889206886292, 0...
| دوست	| [0.07587642222642899, -0.24341894686222076, 0....
| دارم	| [0.0899219810962677, -0.21863090991973877, 0.4...
```
{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|persian_w2v_cc_300d|
|Type:|embeddings|
|Compatibility:|Spark NLP 2.7.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[word_embeddings]|
|Language:|fa|
|Case sensitive:|false|
|Dimension:|300|

## Data Source

This model is imported from https://fasttext.cc/docs/en/crawl-vectors.html