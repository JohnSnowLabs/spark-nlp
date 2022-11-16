---
layout: model
title: ELECTRA Sentence Embeddings(ELECTRA Large)
author: John Snow Labs
name: sent_electra_large_uncased
date: 2020-08-27
task: Embeddings
language: en
edition: Spark NLP 2.6.0
spark_version: 2.4
tags: [open_source, embeddings, en]
supported: true
annotator: BertSentenceEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
ELECTRA is a BERT-like model that is pre-trained as a discriminator in a set-up resembling a generative adversarial network (GAN). It was originally published by:
Kevin Clark and Minh-Thang Luong and Quoc V. Le and Christopher D. Manning: [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://openreview.net/forum?id=r1xMH1BtvB), ICLR 2020.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_electra_large_uncased_en_2.6.0_2.4_1598489955147.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
embeddings = BertSentenceEmbeddings.pretrained("sent_electra_large_uncased", "en") \
.setInputCols("sentence") \
.setOutputCol("sentence_embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, embeddings])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
result = pipeline_model.transform(spark.createDataFrame([['I hate cancer', "Antibiotics aren't painkiller"]], ["text"]))
```

```scala
...
val embeddings = BertSentenceEmbeddings.pretrained("sent_electra_large_uncased", "en")
.setInputCols("sentence")
.setOutputCol("sentence_embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, embeddings))
val data = Seq("I hate cancer, "Antibiotics aren't painkiller").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["I hate cancer", "Antibiotics aren't painkiller"]
embeddings_df = nlu.load('en.embed_sentence.electra_large_uncased').predict(text, output_level='sentence')
embeddings_df
```

</div>

{:.h2_title}
## Results
```bash
	sentence	                        en_embed_sentence_electra_large_uncased_embeddings
		
I hate cancer 	                  [-0.5168956518173218, -0.4284093976020813, -0....
Antibiotics aren't painkiller 	[0.03924501687288284, 0.28086787462234497, 0.3...
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_electra_large_uncased|
|Type:|embeddings|
|Compatibility:|Spark NLP 2.6.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[sentence_embeddings]|
|Language:|[en]|
|Dimension:|1024|
|Case sensitive:|false|

{:.h2_title}
## Data Source
The model is imported from https://tfhub.dev/google/electra_large/2
