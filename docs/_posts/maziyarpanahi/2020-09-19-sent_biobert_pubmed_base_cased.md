---
layout: model
title: BioBERT Sentence Embeddings (Pubmed)
author: John Snow Labs
name: sent_biobert_pubmed_base_cased
date: 2020-09-19
task: Embeddings
language: en
edition: Spark NLP 2.6.2
spark_version: 2.4
tags: [embeddings, en, open_source]
supported: true
annotator: BertSentenceEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
This model contains a pre-trained weights of BioBERT, a language representation model for biomedical domain, especially designed for biomedical text mining tasks such as biomedical named entity recognition, relation extraction, question answering, etc. The details are described in the paper "[BioBERT: a pre-trained biomedical language representation model for biomedical text mining](https://arxiv.org/abs/1901.08746)".

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_biobert_pubmed_base_cased_en_2.6.2_2.4_1600449483871.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
embeddings = BertSentenceEmbeddings.pretrained("sent_biobert_pubmed_base_cased", "en") \
.setInputCols("sentence") \
.setOutputCol("sentence_embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, embeddings])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
result = pipeline_model.transform(spark.createDataFrame([['I hate cancer', "Antibiotics aren't painkiller"]], ["text"]))
```

```scala
...
val embeddings = BertSentenceEmbeddings.pretrained("sent_biobert_pubmed_base_cased", "en")
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
embeddings_df = nlu.load('en.embed_sentence.biobert.pubmed_base_cased').predict(text, output_level='sentence')
embeddings_df
```

</div>

{:.h2_title}
## Results
```bash
en_embed_sentence_biobert_pubmed_base_cased_embeddings	sentence

	[0.209750697016716, 0.21535921096801758, -0.59...	I hate cancer
	[0.01466107927262783, -0.20778851211071014, -0...	Antibiotics aren't painkiller
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_biobert_pubmed_base_cased|
|Type:|embeddings|
|Compatibility:|Spark NLP 2.6.2|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[sentence_embeddings]|
|Language:|[en]|
|Dimension:|768|
|Case sensitive:|true|


{:.h2_title}
## Data Source
The model is imported from [https://github.com/dmis-lab/biobert](https://github.com/dmis-lab/biobert)
