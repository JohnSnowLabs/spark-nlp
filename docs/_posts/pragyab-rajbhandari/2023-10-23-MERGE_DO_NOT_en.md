---
layout: model
title: FOR_TESTING_PURPOSE
author: John Snow Labs
name: MERGE_DO_NOT
date: 2023-10-23
tags: [en, open_source, tensorflow]
task: Named Entity Recognition
language: en
edition: Spark NLP 4.1.0
spark_version: 3.2
supported: true
engine: tensorflow
annotator: NerDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

THIS IS A TEST

## Predicted Entities

`MedicalCondition`, `Pathogen`, `Medicine`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/MERGE_DO_NOT_en_4.1.0_3.2_1698079467463.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/MERGE_DO_NOT_en_4.1.0_3.2_1698079467463.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()
			.setInputCol("text")
			.setOutputCol("document")

sentence_detector = SentenceDetector()
			.setInputCols(["document"])
			.setOutputCol("sentence")
			.setCustomBounds([""])

tokenizer = Tokenizer()
		.setInputCols(["sentence"])
		.setOutputCol(\"token\")
		.setSplitChars(['-'])"

word_embeddings = WordEmbeddingsModel()
			.pretrained("glove_100d", "en", "clinical/models")
			.setInputCols(["sentence", "token"])
			.setOutputCol("embeddings")

ner = NerDLModel().pretrained("ner_NER_TEST_TRAINING", "en","clinical/models")
		.setInputCols(["sentence", "token", "embeddings"])
		.setOutputCol("ner")

ner_converter = NerConverter()
			.setInputCols(["sentence", "token", "ner"])
			.setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[document_assembler,
			    sentence_detector,
			    tokenizer,
			    word_embeddings,
			    ner,
			    ner_converter])

data = spark.createDataFrame([["SAMPLE_TEXT"]]).toDF("text")
result = pipeline.fit(data).transform(data)
```

</div>

## Results

```bash
TEST
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|MERGE_DO_NOT|
|Type:|ner|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|2.5 MB|
|Dependencies:|glove_100d|