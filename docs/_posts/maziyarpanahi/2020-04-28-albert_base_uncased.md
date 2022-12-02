---
layout: model
title: ALBERT Embeddings (Base Uncase)
author: John Snow Labs
name: albert_base_uncased
date: 2020-04-28
task: Embeddings
language: en
edition: Spark NLP 2.5.0
spark_version: 2.4
tags: [embeddings, en, open_source]
supported: true
annotator: AlBertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
ALBERT is "A Lite" version of BERT, a popular unsupervised language representation learning algorithm. ALBERT uses parameter-reduction techniques that allow for large-scale configurations, overcome previous memory limitations, and achieve better behavior with respect to model degradation. The details are described in the paper "[ALBERT: A Lite BERT for Self-supervised Learning of Language Representations.](https://arxiv.org/abs/1909.11942)"

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_base_uncased_en_2.5.0_2.4_1588073363475.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
embeddings = AlbertEmbeddings.pretrained("albert_base_uncased", "en") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
result = pipeline_model.transform(spark.createDataFrame([['I love NLP']], ["text"]))
```

```scala
...
val embeddings = AlbertEmbeddings.pretrained("albert_base_uncased", "en")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
val data = Seq("I love NLP").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["I love NLP"]
embeddings_df = nlu.load('en.embed.albert.base_uncased').predict(text, output_level='token')
embeddings_df
```

</div>

{:.h2_title}
## Results
```bash
token	en_embed_albert_base_uncased_embeddings
	
	I	[1.0153148174285889, 0.5481745600700378, -0.44...
	love	[0.3452114760875702, -1.191628336906433, 0.423...
	NLP	[-0.4268064796924591, -0.3819553852081299, 0.8...
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|albert_base_uncased|
|Type:|embeddings|
|Compatibility:|Spark NLP 2.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[word_embeddings]|
|Language:|[en]|
|Dimension:|768|
|Case sensitive:|false|


{:.h2_title}
## Data Source
The model is imported from [https://tfhub.dev/google/albert_base/3](https://tfhub.dev/google/albert_base/3)
