---
layout: model
title: Finnish BERT Embeddings (Base Uncased)
author: John Snow Labs
name: bert_finnish_uncased
date: 2020-08-31
task: Embeddings
language: fi
edition: Spark NLP 2.6.0
spark_version: 2.4
tags: [open_source, embeddings, fi]
supported: true
deprecated: true
annotator: BertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
A version of Google's BERT deep transfer learning model for Finnish. The model can be fine-tuned to achieve state-of-the-art results for various Finnish natural language processing tasks. `FinBERT` features a custom 50,000 wordpiece vocabulary that has much better coverage of Finnish words.

`FinBERT` has been pre-trained for 1 million steps on over 3 billion tokens (24B characters) of Finnish text drawn from news, online discussion, and internet crawls. By contrast, Multilingual BERT was trained on Wikipedia texts, where the Finnish Wikipedia text is approximately 3% of the amount used to train `FinBERT`.

These features allow `FinBERT` to outperform not only Multilingual BERT but also all previously proposed models when fine-tuned for Finnish natural language processing tasks.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_finnish_uncased_fi_2.6.0_2.4_1598897239983.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
embeddings = BertEmbeddings.pretrained("bert_finnish_uncased", "fi") \
      .setInputCols("sentence", "token") \
      .setOutputCol("embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
result = pipeline_model.transform(spark.createDataFrame([['Rakastan NLP: tä']], ["text"]))
```

```scala
...
val embeddings = BertEmbeddings.pretrained("bert_finnish_uncased", "fi")
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
val data = Seq("Rakastan NLP: tä").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["Rakastan NLP: tä"]
embeddings_df = nlu.load('fi.embed.bert.uncased.').predict(text, output_level='token')
embeddings_df
```

</div>

{:.h2_title}
## Results
```bash
	token	    fi_embed_bert_uncased__embeddings
		
      Rakastan  [-0.5126021504402161, -1.1741008758544922, 0.6...
 	NLP 	    [1.4763829708099365, -1.5427947044372559, 0.80...
 	: 	    [-0.2581554353237152, -0.5670831203460693, -1....
 	tä 	    [0.39770740270614624, -0.7221324443817139, 0.1...
```


{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_finnish_uncased|
|Type:|embeddings|
|Compatibility:|Spark NLP 2.6.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[word_embeddings]|
|Language:|[fi]|
|Dimension:|768|
|Case sensitive:|false|

{:.h2_title}
## Data Source
The model is imported from https://github.com/TurkuNLP/FinBERT
