---
layout: model
title: COVID BERT Embeddings (Large Uncased)
author: John Snow Labs
name: covidbert_large_uncased
date: 2020-08-27
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
BERT-large-uncased model, pretrained on a corpus of messages from Twitter about COVID-19. 

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/covidbert_large_uncased_en_2.6.0_2.4_1598484981419.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/covidbert_large_uncased_en_2.6.0_2.4_1598484981419.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
embeddings = BertEmbeddings.pretrained("covidbert_large_uncased", "en") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
result = pipeline_model.transform(spark.createDataFrame([['I love NLP']], ["text"]))
```

```scala
...
val embeddings = BertEmbeddings.pretrained("covidbert_large_uncased", "en")
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
embeddings_df = nlu.load('en.embed.covidbert.large_uncased').predict(text, output_level='token')
embeddings_df
```

</div>

{:.h2_title}
## Results
```bash
	en_embed_covidbert_large_uncased_embeddings	      token
	
[-1.934066891670227, 0.620597779750824, 0.0967... 	I
[-0.5530431866645813, 1.1948248147964478, -0.0... 	love
[0.255395770072937, 0.5808677077293396, 0.3073... 	NLP
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|covidbert_large_uncased|
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
The model is imported from https://tfhub.dev/digitalepidemiologylab/covid-twitter-bert/2
