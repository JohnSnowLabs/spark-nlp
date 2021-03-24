---
layout: model
title: BioBERT Sentence Embeddings (Pubmed PMC)
author: John Snow Labs
name: sent_biobert_pubmed_pmc_base_cased
date: 2020-08-25
task: Embeddings
language: en
edition: Spark NLP 2.6.0
tags: [embeddings, en, open_source]
supported: false
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
This model contains a pre-trained weights of BioBERT, a language representation model for biomedical domain, especially designed for biomedical text mining tasks such as biomedical named entity recognition, relation extraction, question answering, etc. The details are described in the paper "[BioBERT: a pre-trained biomedical language representation model for biomedical text mining](https://arxiv.org/abs/1901.08746)".

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_biobert_pubmed_pmc_base_cased_en_2.6.0_2.4_1598349155555.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
embeddings = BertSentenceEmbeddings.pretrained("sent_biobert_pubmed_pmc_base_cased", "en") \
      .setInputCols("sentence") \
      .setOutputCol("sentence_embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, embeddings])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
result = pipeline_model.transform(spark.createDataFrame(pd.DataFrame({"text": ["I hate cancer, "Antibiotics aren't painkiller"]})))
```

```scala
...
val embeddings = BertSentenceEmbeddings.pretrained("sent_biobert_pubmed_pmc_base_cased", "en")
      .setInputCols("sentence")
      .setOutputCol("sentence_embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, embeddings))
val result = pipeline.fit(Seq.empty["I hate cancer, "Antibiotics aren't painkiller"].toDS.toDF("text")).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["I hate cancer", "Antibiotics aren't painkiller"]
embeddings_df = nlu.load('en.embed_sentence.biobert.pubmed_pmc_base_cased').predict(text, output_level='sentence')
embeddings_df
```

</div>

{:.h2_title}
## Results
```bash
      sentence	                en_embed_sentence_biobert_pubmed_pmc_base_cased_embeddings

	I hate cancer	                [0.2354733943939209, 0.30127033591270447, -0.1...
	Antibiotics aren't painkiller	    [0.2837969958782196, 0.03842488303780556, 0.04...
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_biobert_pubmed_pmc_base_cased|
|Type:|embeddings|
|Compatibility:|Spark NLP 2.6.0|
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
