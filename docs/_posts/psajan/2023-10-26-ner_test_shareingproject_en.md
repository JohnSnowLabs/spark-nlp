---
layout: model
title: test publishing model
author: John Snow Labs
name: ner_test_shareingproject
date: 2023-10-26
tags: [en, open_source, tensorflow]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.1.0
spark_version: 3.2
supported: true
engine: tensorflow
annotator: NerDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

test do not publish

## Predicted Entities

`Pathogen`, `MedicalCondition`, `Medicine`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_test_shareingproject_en_5.1.0_3.2_1698314762801.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ner_test_shareingproject_en_5.1.0_3.2_1698314762801.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

ner = NerDLModel().pretrained("ner_test-kc-shareingproject", "en","clinical/models")
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

data = spark.createDataFrame([["Sample text"]]).toDF("text")
result = pipeline.fit(data).transform(data)
```

</div>

## Results

```bash
Empty DataFrame
Columns: [chunk, ner_label]
Index: []
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_test_shareingproject|
|Type:|ner|
|Compatibility:|Spark NLP 5.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|2.5 MB|
|Dependencies:|glove_100d|