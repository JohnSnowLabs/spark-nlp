---
layout: model
title: BERT Embeddings (Large Uncased)
author: John Snow Labs
name: bert_large_uncased
date: 2020-08-25
task: Embeddings
language: en
edition: Spark NLP 2.6.0
spark_version: 2.4
tags: [open_source, embeddings, en]
supported: true
annotator: BertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
This model contains a deep bidirectional transformer trained on Wikipedia and the BookCorpus. The details are described in the paper "[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)".

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_large_uncased_en_2.6.0_2.4_1598341287005.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
embeddings = BertEmbeddings.pretrained("bert_large_uncased", "en") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
result = pipeline_model.transform(spark.createDataFrame([['I love NLP']], ["text"]))
```

```scala
...
val embeddings = BertEmbeddings.pretrained("bert_large_uncased", "en")
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
embeddings_df = nlu.load('en.embed.bert.large_uncased').predict(text, output_level='token')
embeddings_df
```

</div>

{:.h2_title}
## Results
```bash
	en_embed_bert_large_uncased_embeddings	token
	
	[-0.07447264343500137, -0.337308406829834, -0....	I
	[-0.5735481977462769, -0.3580206632614136, -0....	love
	[-0.3929762840270996, -0.4147087037563324, 0.2...	NLP
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_large_uncased|
|Type:|embeddings|
|Compatibility:|Spark NLP 2.6.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[word_embeddings]|
|Language:|[en]|
|Dimension:|1024|
|Case sensitive:|false|

{:.h2_title}
## Data Source
The model is imported from [https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1](https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1)
