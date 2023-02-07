---
layout: model
title: Portuguese BERT Embeddings (Large Cased)
author: John Snow Labs
name: bert_portuguese_large_cased
date: 2020-11-04
task: Embeddings
language: pt
edition: Spark NLP 2.6.0
spark_version: 2.4
tags: [open_source, embeddings, pt]
supported: true
annotator: BertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
This is the pre-trained BERT model trained on the Portuguese language. `BERT-Base` and `BERT-Large` Cased variants were trained on the `BrWaC` (Brazilian Web as Corpus), a large Portuguese corpus, for 1,000,000 steps, using whole-word mask.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_portuguese_large_cased_pt_2.6.0_2.4_1604487922125.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_portuguese_large_cased_pt_2.6.0_2.4_1604487922125.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
embeddings = BertEmbeddings.pretrained("bert_portuguese_large_cased", "pt") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
result = pipeline_model.transform(spark.createDataFrame([['Eu amo PNL']], ["text"]))
```

```scala
...
val embeddings = BertEmbeddings.pretrained("bert_portuguese_large_cased", "pt")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
val data = Seq("Eu amo PNL").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["Eu amo PNL"]
embeddings_df = nlu.load('pt.bert.cased.large').predict(text, output_level='token')
embeddings_df
```

</div>

{:.h2_title}
## Results
```bash
	token	pt_bert_cased_large_embeddings
		
Eu 	[0.6893012523651123, 0.18436528742313385, 0.14...
	amo 	[0.6536692976951599, 0.17582201957702637, -0.5...
	PNL 	[-0.1397203803062439, 0.5698696374893188, -0.3...
```


{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_portuguese_large_cased|
|Type:|embeddings|
|Compatibility:|Spark NLP 2.6.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[word_embeddings]|
|Language:|[pt]|
|Dimension:|1024|
|Case sensitive:|true|

{:.h2_title}
## Data Source
The model is imported from https://github.com/neuralmind-ai/portuguese-bert
