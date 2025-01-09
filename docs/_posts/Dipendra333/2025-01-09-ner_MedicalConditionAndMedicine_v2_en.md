---
layout: model
title: For testing only.Deny the PR.
author: John Snow Labs
name: ner_MedicalConditionAndMedicine_v2
date: 2025-01-09
tags: [en, open_source, tensorflow]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.4.1
spark_version: 3.4
supported: true
engine: tensorflow
annotator: NerDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

For regression testing only. Please deny the PR.

## Predicted Entities

`Pathogen`, `MedicalCondition`, `Medicine`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_MedicalConditionAndMedicine_v2_en_5.4.1_3.4_1736414292917.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ner_MedicalConditionAndMedicine_v2_en_5.4.1_3.4_1736414292917.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
			.pretrained("glove_100d", "en" )
			.setInputCols(["sentence", "token"])
			.setOutputCol("embeddings")

ner = NerDLModel().pretrained("ner_MedicalConditionAndMedicine_v2", "en" )
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
For regression testing only. Please deny the PR.
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_MedicalConditionAndMedicine_v2|
|Type:|ner|
|Compatibility:|Spark NLP 5.4.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|2.5 MB|
|Dependencies:|glove_100d|