---
layout: model
title: BioBERT Embeddings (Discharge)
author: John Snow Labs
name: biobert_discharge_base_cased
date: 2020-09-19
task: Embeddings
language: en
edition: Spark NLP 2.6.2
spark_version: 2.4
tags: [embeddings, en, open_source]
supported: true
annotator: BertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
This model contains a pre-trained weights of ClinicalBERT for discharge summaries. This domain-specific model has performance improvements on 3/5 clinical NLP tasks andd establishing a new state-of-the-art on the MedNLI dataset. The details are described in the paper "[Publicly Available Clinical BERT Embeddings](https://www.aclweb.org/anthology/W19-1909/)".

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/biobert_discharge_base_cased_en_2.6.2_2.4_1600531401858.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
embeddings = BertEmbeddings.pretrained("biobert_discharge_base_cased", "en") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
result = pipeline_model.transform(spark.createDataFrame([['I hate cancer']], ["text"]))
```

```scala
...
val embeddings = BertEmbeddings.pretrained("biobert_discharge_base_cased", "en")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
val data = Seq("I hate cancer").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["I hate cancer"]
embeddings_df = nlu.load('en.embed.biobert.discharge_base_cased').predict(text, output_level='token')
embeddings_df
```

</div>

{:.h2_title}
## Results
```bash
token	en_embed_biobert_discharge_base_cased_embeddings

	I	[0.0036486536264419556, 0.3796533942222595, -0...
	hate	[0.1914958357810974, 0.6709488034248352, -0.49...
	cancer	[0.04618441313505173, -0.04562612622976303, -0...
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|biobert_discharge_base_cased|
|Type:|embeddings|
|Compatibility:|Spark NLP 2.6.2|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[word_embeddings]|
|Language:|[en]|
|Dimension:|768|
|Case sensitive:|true|


{:.h2_title}
## Data Source
The model is imported from [https://github.com/EmilyAlsentzer/clinicalBERT](https://github.com/EmilyAlsentzer/clinicalBERT)
