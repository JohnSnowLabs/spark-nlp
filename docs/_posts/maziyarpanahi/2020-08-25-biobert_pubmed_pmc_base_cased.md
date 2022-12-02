---
layout: model
title: BioBERT Embeddings (Pubmed PMC)
author: John Snow Labs
name: biobert_pubmed_pmc_base_cased
date: 2020-08-25
task: Embeddings
language: en
edition: Spark NLP 2.6.0
spark_version: 2.4
tags: [embeddings, en, open_source]
supported: true
annotator: BertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
This model contains a pre-trained weights of BioBERT, a language representation model for biomedical domain, especially designed for biomedical text mining tasks such as biomedical named entity recognition, relation extraction, question answering, etc. The details are described in the paper "[BioBERT: a pre-trained biomedical language representation model for biomedical text mining](https://arxiv.org/abs/1901.08746)".

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/biobert_pubmed_pmc_base_cased_en_2.6.0_2.4_1598343200280.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
embeddings = BertEmbeddings.pretrained("biobert_pubmed_pmc_base_cased", "en") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
result = pipeline_model.transform(spark.createDataFrame([['I hate cancer']], ["text"]))
```

```scala
...
val embeddings = BertEmbeddings.pretrained("biobert_pubmed_pmc_base_cased", "en")
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
embeddings_df = nlu.load('en.embed.biobert.pubmed_pmc_base_cased').predict(text, output_level='token')
embeddings_df
```

</div>

{:.h2_title}
## Results
```bash
token	en_embed_biobert_pubmed_pmc_base_cased_embeddings

	I	[-0.012962102890014648, 0.27699071168899536, 0...
	hate	[0.1688309609889984, 0.5337603688240051, 0.148...
	cancer	[0.1850549429655075, 0.05875205248594284, -0.5...
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|biobert_pubmed_pmc_base_cased|
|Type:|embeddings|
|Compatibility:|Spark NLP 2.6.0|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[word_embeddings]|
|Language:|[en]|
|Dimension:|768|
|Case sensitive:|true|


{:.h2_title}
## Data Source
The model is imported from [https://github.com/dmis-lab/biobert](https://github.com/dmis-lab/biobert)
