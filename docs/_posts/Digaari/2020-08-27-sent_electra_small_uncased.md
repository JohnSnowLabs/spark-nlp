---
layout: model
title: ELECTRA Sentence Embeddings(ELECTRA Small)
author: John Snow Labs
name: sent_electra_small_uncased
date: 2020-08-27
task: Embeddings
language: en
edition: Spark NLP 2.6.0
tags: [open_source, embeddings, en]
supported: false
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_electra_small_uncased_en_2.6.0_2.4_1598489761661.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
embeddings = BertEmbeddings.pretrained("sent_electra_small_uncased", "en") \
      .setInputCols("sentence") \
      .setOutputCol("sentence_embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, embeddings])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
result = pipeline_model.transform(spark.createDataFrame(pd.DataFrame({"text": ["I hate cancer, "Antibiotics aren't painkiller"]})))
```

```scala
...
val embeddings = BertEmbeddings.pretrained("sent_electra_small_uncased", "en")
      .setInputCols("sentence")
      .setOutputCol("sentence_embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, embeddings))
val result = pipeline.fit(Seq.empty["I hate cancer, "Antibiotics aren't painkiller"].toDS.toDF("text")).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["I hate cancer", "Antibiotics aren't painkiller"]
embeddings_df = nlu.load('en.embed_sentence.electra_small_uncased').predict(text, output_level='sentence')
embeddings_df
```

</div>

{:.h2_title}
## Results
```bash
	sentence	                        en_embed_sentence_electra_small_uncased_embeddings
		
      I hate cancer 	                  [0.4288138449192047, -0.25909560918807983, -0....
 	Antibiotics aren't painkiller 	[0.04786013811826706, 0.14878112077713013, -0....
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_electra_small_uncased|
|Type:|embeddings|
|Compatibility:|Spark NLP 2.6.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[sentence_embeddings]|
|Language:|[en]|
|Dimension:|256|
|Case sensitive:|false|

{:.h2_title}
## Data Source
The model is imported from [https://tfhub.dev/google/electra_small/2](https://tfhub.dev/google/electra_small/2)
