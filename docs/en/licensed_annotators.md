---
layout: docs
header: true
title: Spark NLP for Healthcare Annotators
permalink: /docs/en/licensed_annotators
key: docs-licensed-annotators
modify_date: "2020-08-10"
use_language_switcher: "Python-Scala"
---

<div class="h3-box" markdown="1">

A Spark NLP for Healthcare subscription includes access to several pretrained annotators.
Check out www.johnsnowlabs.com for more information.

</div><div class="h3-box" markdown="1">

### AssertionLogReg
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/assertion/logreg/AssertionLogRegApproach.html">Estimator scaladocs</a> |
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/assertion/logreg/AssertionLogRegModel.html">Transformer scaladocs</a>

This annotator classifies each clinically relevant named entity into its assertion:

type: "present", "absent", "hypothetical", "conditional", "associated_with_other_person", etc.

**Input types:** `DOCUMENT, CHUNK, WORD_EMBEDDINGS`

**Output type:** `ASSERTION`

<details>
<summary><b>Show Example </b></summary>

<div class="tabs-box" markdown="1">

  {% include programmingLanguageSelectScalaPython.html %}

  ```scala
  // Training with Glove Embeddings
  // First define pipeline stages to extract embeddings and text chunks
  val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

  val glove = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols("document", "token")
    .setOutputCol("word_embeddings")
    .setCaseSensitive(false)

  val chunkAssembler = new Doc2Chunk()
    .setInputCols("document")
    .setChunkCol("target")
    .setOutputCol("chunk")

  // Then the AssertionLogRegApproach model is defined. Label column is needed in the dataset for training.
  val assertion = new AssertionLogRegApproach()
    .setLabelCol("label")
    .setInputCols("document", "chunk", "word_embeddings")
    .setOutputCol("assertion")
    .setReg(0.01)
    .setBefore(11)
    .setAfter(13)
    .setStartCol("start")
    .setEndCol("end")

  val assertionPipeline = new Pipeline().setStages(Array(
    documentAssembler,
    sentenceDetector,
    tokenizer,
    embeddings,
    nerModel,
    nerConverter,
    assertion
  ))

  val assertionModel = assertionPipeline.fit(dataset)

  ```
  ```python
  # Training with Glove Embeddings
  # First define pipeline stages to extract embeddings and text chunks
  documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

  tokenizer = Tokenizer()
    .setInputCols(["document"])
    .setOutputCol("token")

  glove = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("word_embeddings") \
    .setCaseSensitive(False)

  chunkAssembler = Doc2Chunk() \
    .setInputCols(["document"]) \
    .setChunkCol("target") \
    .setOutputCol("chunk")

  # Then the AssertionLogRegApproach model is defined. Label column is needed in the dataset for training.
  assertion = AssertionLogRegApproach() \
    .setLabelCol("label") \
    .setInputCols(["document", "chunk", "word_embeddings"]) \
    .setOutputCol("assertion") \
    .setReg(0.01) \
    .setBefore(11) \
    .setAfter(13) \
    .setStartCol("start") \
    .setEndCol("end")

  assertionPipeline = Pipeline(stages=[
    documentAssembler,
    sentenceDetector,
    tokenizer,
    embeddings,
    nerModel,
    nerConverter,
    assertion
  ])

  assertionModel = assertionPipeline.fit(dataset)
  ```

</div>

</details>

</div>

<div class="h3-box" markdown="1">

### AssertionDL
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/assertion/dl/AssertionDLApproach.html">Estimator scaladocs</a> |
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/assertion/dl/AssertionDLModel.html">Transformer scaladocs</a>

A Deep Learning based approach is used to extract Assertion Status from extracted entities and text. AssertionDLModel requires DOCUMENT, CHUNK and WORD_EMBEDDINGS type annotator inputs, which can be obtained by e.g a DocumentAssembler, NerConverter and WordEmbeddingsModel. The result is an assertion status annotation for each recognized entity. Possible values include `“present”, “absent”, “hypothetical”, “conditional”, “associated_with_other_person”` etc. For pretrained models please see the Models Hub for available models.

**Input types:** `DOCUMENT, CHUNK, WORD_EMBEDDINGS`

**Output type:** `ASSERTION`

<details>
<summary><b>Show Example</b></summary>

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```scala
// Define pipeline stages to extract NER chunks first
val data = Seq(
  "Patient with severe fever and sore throat",
  "Patient shows no stomach pain",
  "She was maintained on an epidural and PCA for pain control.").toDF("text")
val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
val sentenceDetector = new SentenceDetector().setInputCols("document").setOutputCol("sentence")
val tokenizer = new Tokenizer().setInputCols("sentence").setOutputCol("token")
val embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setOutputCol("embeddings")
val nerModel = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models")
  .setInputCols("sentence", "token", "embeddings").setOutputCol("ner")
val nerConverter = new NerConverter().setInputCols("sentence", "token", "ner").setOutputCol("ner_chunk")

// Then a pretrained AssertionDLModel is used to extract the assertion status
val clinicalAssertion = AssertionDLModel.pretrained("assertion_dl", "en", "clinical/models")
  .setInputCols("sentence", "ner_chunk", "embeddings")
  .setOutputCol("assertion")

val assertionPipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings,
  nerModel,
  nerConverter,
  clinicalAssertion
))

val assertionModel = assertionPipeline.fit(data)

// Show results
val result = assertionModel.transform(data)
result.selectExpr("ner_chunk.result", "assertion.result").show(3, truncate=false)
+--------------------------------+--------------------------------+
|result                          |result                          |
+--------------------------------+--------------------------------+
|[severe fever, sore throat]     |[present, present]              |
|[stomach pain]                  |[absent]                        |
|[an epidural, PCA, pain control]|[present, present, hypothetical]|
+--------------------------------+--------------------------------+

```
```python
# Define pipeline stages to extract NER chunks first
data = spark.createDataFrame([
  "Patient with severe fever and sore throat",
  "Patient shows no stomach pain",
  "She was maintained on an epidural and PCA for pain control."]).toDF("text")
documentAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
sentenceDetector = SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
tokenizer = Tokenizer().setInputCols(["sentence"]).setOutputCol("token")
embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models") \
  .setOutputCol("embeddings")
nerModel = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]).setOutputCol("ner")
nerConverter = NerConverter().setInputCols(["sentence", "token", "ner"]).setOutputCol("ner_chunk")

# Then a pretrained AssertionDLModel is used to extract the assertion status
clinicalAssertion = AssertionDLModel.pretrained("assertion_dl", "en", "clinical/models") \
  .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
  .setOutputCol("assertion")

assertionPipeline = Pipeline(stages=[
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings,
  nerModel,
  nerConverter,
  clinicalAssertion
])

assertionModel = assertionPipeline.fit(data)

# Show results
result = assertionModel.transform(data)
result.selectExpr("ner_chunk.result", "assertion.result").show(3, truncate=False)
+--------------------------------+--------------------------------+
|result                          |result                          |
+--------------------------------+--------------------------------+
|[severe fever, sore throat]     |[present, present]              |
|[stomach pain]                  |[absent]                        |
|[an epidural, PCA, pain control]|[present, present, hypothetical]|
+--------------------------------+--------------------------------+

```

</div>

</details>

</div>

<div class="h3-box" markdown="1">
### AssertionFilterer

<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/chunker/AssertionFilterer.html">Transformer scaladocs</a>

Filters entities coming from ASSERTION type annotations and returns the CHUNKS. Filters can be set via a white list on the extracted chunk, the assertion or a regular expression. White list for assertion is enabled by default. To use chunk white list, `criteria` has to be set to `"isin"`. For regex, `criteria` has to be set to `"regex"`.

**Input types:** `DOCUMENT, CHUNK, ASSERTION`

**Output type:** `CHUNK`

<details>
<summary><b>Show Example</b></summary>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}

```scala
// To see how the assertions are extracted, see the example for AssertionDLModel.
// Define an extra step where the assertions are filtered
val assertionFilterer = new AssertionFilterer()
  .setInputCols("sentence","ner_chunk","assertion")
  .setOutputCol("filtered")
  .setCriteria("assertion")
  .setWhiteList("present")

val assertionPipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings,
  nerModel,
  nerConverter,
  clinicalAssertion,
  assertionFilterer
))

val assertionModel = assertionPipeline.fit(data)
val result = assertionModel.transform(data)

// Show results:
result.selectExpr("ner_chunk.result", "assertion.result").show(3, truncate=false)
+--------------------------------+--------------------------------+
|result                          |result                          |
+--------------------------------+--------------------------------+
|[severe fever, sore throat]     |[present, present]              |
|[stomach pain]                  |[absent]                        |
|[an epidural, PCA, pain control]|[present, present, hypothetical]|
+--------------------------------+--------------------------------+

result.select("filtered.result").show(3, truncate=false)
+---------------------------+
|result                     |
+---------------------------+
|[severe fever, sore throat]|
|[]                         |
|[an epidural, PCA]         |
+---------------------------+

```
```python
# To see how the assertions are extracted, see the example for AssertionDLModel.
# Define an extra step where the assertions are filtered
assertionFilterer = AssertionFilterer() \
  .setInputCols(["sentence","ner_chunk","assertion"]) \
  .setOutputCol("filtered") \
  .setCriteria("assertion") \
  .setWhiteList("present")

assertionPipeline = Pipeline(stages=[
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings,
  nerModel,
  nerConverter,
  clinicalAssertion,
  assertionFilterer
])

assertionModel = assertionPipeline.fit(data)
result = assertionModel.transform(data)

# Show results:
result.selectExpr("ner_chunk.result", "assertion.result").show(3, truncate=False)
+--------------------------------+--------------------------------+
|result                          |result                          |
+--------------------------------+--------------------------------+
|[severe fever, sore throat]     |[present, present]              |
|[stomach pain]                  |[absent]                        |
|[an epidural, PCA, pain control]|[present, present, hypothetical]|
+--------------------------------+--------------------------------+

result.select("filtered.result").show(3, truncate=False)
+---------------------------+
|result                     |
+---------------------------+
|[severe fever, sore throat]|
|[]                         |
|[an epidural, PCA]         |
+---------------------------+

```
</div>

</details>

</div>

<div class="h3-box" markdown="1">

### Chunk2Token
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/Chunk2Token.html">Transformer scaladocs</a>

Transforms a complete chunk Annotation into a token Annotation without further tokenization, as opposed to ChunkTokenizer.

**Input types:** `CHUNK,`

**Output type:** `TOKEN`

**Example:**

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
chunk2Token = Chunk2Token() \
    .setInputCols(["chunk"]) \
    .setOutputCol("token")
```
```scala
val chunk2Token = new Chunk2Token()
    .setInputCols("chunk")
    .setOutputCol("token")
```

</div></div><div class="h3-box" markdown="1">

### ChunkEntityResolver
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/resolution/ChunkEntityResolverApproach.html">Estimator scaladocs</a> |
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/resolution/ChunkEntityResolverModel.html">Transformer scaladocs</a>

Contains all the parameters and methods to train a ChunkEntityResolverModel. It transform a dataset with two Input Annotations of types TOKEN and WORD_EMBEDDINGS, coming from e.g. ChunkTokenizer and ChunkEmbeddings Annotators and returns the normalized entity for a particular trained ontology / curated dataset. (e.g. ICD-10, RxNorm, SNOMED etc.) To use pretrained models please use ChunkEntityResolverModel and see the Models Hub for available models.

**Input types:** `TOKEN, WORD_EMBEDDINGS`

**Output type:** `ENTITY`

<details>
<summary><b>Show Example </b></summary>


<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```scala
// Training a SNOMED model
// Define pre-processing pipeline for training data. It needs consists of columns for the normalized training data
// and their labels.
val document = new DocumentAssembler()
  .setInputCol("normalized_text")
  .setOutputCol("document")

val chunk = new Doc2Chunk()
  .setInputCols("document")
  .setOutputCol("chunk")

val token = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val embeddings = WordEmbeddingsModel.pretrained("embeddings_healthcare_100d", "en", "clinical/models")
  .setInputCols("document", "token")
  .setOutputCol("embeddings")

val chunkEmb = new ChunkEmbeddings()
      .setInputCols("chunk", "embeddings")
      .setOutputCol("chunk_embeddings")

val snomedTrainingPipeline = new Pipeline().setStages(Array(
  document,
  chunk,
  token,
  embeddings,
  chunkEmb
))

val snomedTrainingModel = snomedTrainingPipeline.fit(data)

val snomedData = snomedTrainingModel.transform(data).cache()

// Then the Resolver can be trained with
val snomedExtractor = new ChunkEntityResolverApproach()
  .setInputCols("token", "chunk_embeddings")
  .setOutputCol("recognized")
  .setNeighbours(1000)
  .setAlternatives(25)
  .setNormalizedCol("normalized_text")
  .setLabelCol("label")
  .setEnableWmd(true).setEnableTfidf(true).setEnableJaccard(true)
  .setEnableSorensenDice(true).setEnableJaroWinkler(true).setEnableLevenshtein(true)
  .setDistanceWeights(Array(1, 2, 2, 1, 1, 1))
  .setAllDistancesMetadata(true)
  .setPoolingStrategy("MAX")
  .setThreshold(1e32)
val model = snomedExtractor.fit(snomedData)

```
```python
# Training a SNOMED model
# Define pre-processing pipeline for training data. It needs consists of columns for the normalized training data
# and their labels.
document = DocumentAssembler() \
  .setInputCol("normalized_text") \
  .setOutputCol("document")

chunk = Doc2Chunk() \
  .setInputCols(["document"]) \
  .setOutputCol("chunk")

token = Tokenizer()
  .setInputCols(["document"])
  .setOutputCol("token")

embeddings = WordEmbeddingsModel.pretrained("embeddings_healthcare_100d", "en", "clinical/models") \
  .setInputCols(["document", "token"]) \
  .setOutputCol("embeddings")

chunkEmb = ChunkEmbeddings() \
      .setInputCols(["chunk", "embeddings"]) \
      .setOutputCol("chunk_embeddings")

snomedTrainingPipeline = Pipeline(stages=[
  document,
  chunk,
  token,
  embeddings,
  chunkEmb
])

snomedTrainingModel = snomedTrainingPipeline.fit(data)

snomedData = snomedTrainingModel.transform(data).cache()

# Then the Resolver can be trained with
snomedExtractor = ChunkEntityResolverApproach() \
  .setInputCols(["token", "chunk_embeddings"]) \
  .setOutputCol("recognized") \
  .setNeighbours(1000) \
  .setAlternatives(25) \
  .setNormalizedCol("normalized_text") \
  .setLabelCol("label") \
  .setEnableWmd(True).setEnableTfidf(True).setEnableJaccard(True) \
  .setEnableSorensenDice(True).setEnableJaroWinkler(True).setEnableLevenshtein(True) \
  .setDistanceWeights(Array(1, 2, 2, 1, 1, 1)) \
  .setAllDistancesMetadata(True) \
  .setPoolingStrategy("MAX") \
  .setThreshold(1e32)
model = snomedExtractor.fit(snomedData)

```
</div>

</details>
</div>

<div class="h3-box" markdown="1">
### ChunkFilterer

<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/chunker/ChunkFilterer.html">Transformer scaladocs</a>

Filters entities coming from CHUNK annotations. Filters can be set via a white list of terms or a regular expression. White list criteria is enabled by default. To use regex, `criteria` has to be set to `regex`.

**Input types:** `DOCUMENT,CHUNK`

**Output type:** `CHUNK`

<details>
<summary><b>Show Example </b></summary>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}

```scala
// Filtering POS tags
// First pipeline stages to extract the POS tags are defined
val data = Seq("Has a past history of gastroenteritis and stomach pain, however patient ...").toDF("text")
val docAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
val sentenceDetector = new SentenceDetector().setInputCols("document").setOutputCol("sentence")
val tokenizer = new Tokenizer().setInputCols("sentence").setOutputCol("token")

val posTagger = PerceptronModel.pretrained()
  .setInputCols("sentence", "token")
  .setOutputCol("pos")

val chunker = new Chunker()
  .setInputCols("pos", "sentence")
  .setOutputCol("chunk")
  .setRegexParsers(Array("(<NN>)+"))

// Then the chunks can be filtered via a white list. Here only terms with "gastroenteritis" remain.
val chunkerFilter = new ChunkFilterer()
  .setInputCols("sentence","chunk")
  .setOutputCol("filtered")
  .setCriteria("isin")
  .setWhiteList("gastroenteritis")

val pipeline = new Pipeline().setStages(Array(
  docAssembler,
  sentenceDetector,
  tokenizer,
  posTagger,
  chunker,
  chunkerFilter))

result.selectExpr("explode(chunk)").show(truncate=false)
+---------------------------------------------------------------------------------+
|col                                                                              |
+---------------------------------------------------------------------------------+
|{chunk, 11, 17, history, {sentence -> 0, chunk -> 0}, []}                        |
|{chunk, 22, 36, gastroenteritis, {sentence -> 0, chunk -> 1}, []}                |
|{chunk, 42, 53, stomach pain, {sentence -> 0, chunk -> 2}, []}                   |
|{chunk, 64, 70, patient, {sentence -> 0, chunk -> 3}, []}                        |
|{chunk, 81, 110, stomach pain now.We don't care, {sentence -> 0, chunk -> 4}, []}|
|{chunk, 118, 132, gastroenteritis, {sentence -> 0, chunk -> 5}, []}              |
+---------------------------------------------------------------------------------+

result.selectExpr("explode(filtered)").show(truncate=false)
+-------------------------------------------------------------------+
|col                                                                |
+-------------------------------------------------------------------+
|{chunk, 22, 36, gastroenteritis, {sentence -> 0, chunk -> 1}, []}  |
|{chunk, 118, 132, gastroenteritis, {sentence -> 0, chunk -> 5}, []}|
+-------------------------------------------------------------------+

```
```python
# Filtering POS tags
# First pipeline stages to extract the POS tags are defined
data = spark.createDataFrame(["Has a past history of gastroenteritis and stomach pain, however patient ..."]).toDF("text")
docAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
sentenceDetector = SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
tokenizer = Tokenizer().setInputCols(["sentence"]).setOutputCol("token")

posTagger = PerceptronModel.pretrained() \
  .setInputCols(["sentence", "token"]) \
  .setOutputCol("pos")

chunker = Chunker() \
  .setInputCols(["pos", "sentence"]) \
  .setOutputCol("chunk") \
  .setRegexParsers(Array("(<NN>)+"))

# Then the chunks can be filtered via a white list. Here only terms with "gastroenteritis" remain.
chunkerFilter = ChunkFilterer() \
  .setInputCols(["sentence","chunk"]) \
  .setOutputCol("filtered") \
  .setCriteria("isin") \
  .setWhiteList("gastroenteritis")

pipeline = Pipeline(stages=[
  docAssembler,
  sentenceDetector,
  tokenizer,
  posTagger,
  chunker,
  chunkerFilter])

result.selectExpr("explode(chunk)").show(truncate=False)
+---------------------------------------------------------------------------------+
|col                                                                              |
+---------------------------------------------------------------------------------+
|{chunk, 11, 17, history, {sentence -> 0, chunk -> 0}, []}                        |
|{chunk, 22, 36, gastroenteritis, {sentence -> 0, chunk -> 1}, []}                |
|{chunk, 42, 53, stomach pain, {sentence -> 0, chunk -> 2}, []}                   |
|{chunk, 64, 70, patient, {sentence -> 0, chunk -> 3}, []}                        |
|{chunk, 81, 110, stomach pain now.We don't care, {sentence -> 0, chunk -> 4}, []}|
|{chunk, 118, 132, gastroenteritis, {sentence -> 0, chunk -> 5}, []}              |
+---------------------------------------------------------------------------------+

result.selectExpr("explode(filtered)").show(truncate=False)
+-------------------------------------------------------------------+
|col                                                                |
+-------------------------------------------------------------------+
|{chunk, 22, 36, gastroenteritis, {sentence -> 0, chunk -> 1}, []}  |
|{chunk, 118, 132, gastroenteritis, {sentence -> 0, chunk -> 5}, []}|
+-------------------------------------------------------------------+

```
</div>

</details>

</div>

<div class="h3-box" markdown="1">
### ChunkMerge

<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/merge/ChunkMergeApproach.html">Transformer scaladocs</a>

Merges NER Chunks by prioritizing overlapping indices (chunks with longer lengths and highest information will be kept from each ner model). Labels can be changed by setReplaceDictResource.

**Input types:** `CHUNK, CHUNK`

**Output type:** `CHUNK`

<details>
<summary><b>Show Example </b></summary>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}

```scala
// Define a pipeline with 2 different NER models with a ChunkMergeApproach at the end
val data = Seq(("A 63-year-old man presents to the hospital ...")).toDF("text")
val pipeline = new Pipeline().setStages(Array(
  new DocumentAssembler().setInputCol("text").setOutputCol("document"),
  new SentenceDetector().setInputCols("document").setOutputCol("sentence"),
  new Tokenizer().setInputCols("sentence").setOutputCol("token"),
  WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models").setOutputCol("embs"),
  MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models")
    .setInputCols("sentence", "token", "embs").setOutputCol("jsl_ner"),
  new NerConverter().setInputCols("sentence", "token", "jsl_ner").setOutputCol("jsl_ner_chunk"),
  MedicalNerModel.pretrained("ner_bionlp", "en", "clinical/models")
    .setInputCols("sentence", "token", "embs").setOutputCol("bionlp_ner"),
  new NerConverter().setInputCols("sentence", "token", "bionlp_ner")
    .setOutputCol("bionlp_ner_chunk"),
  new ChunkMergeApproach().setInputCols("jsl_ner_chunk", "bionlp_ner_chunk").setOutputCol("merged_chunk")
))

// Show results
val result = pipeline.fit(data).transform(data).cache()
result.selectExpr("explode(merged_chunk) as a")
  .selectExpr("a.begin","a.end","a.result as chunk","a.metadata.entity as entity")
  .show(5, false)
+-----+---+-----------+---------+
|begin|end|chunk      |entity   |
+-----+---+-----------+---------+
|5    |15 |63-year-old|Age      |
|17   |19 |man        |Gender   |
|64   |72 |recurrent  |Modifier |
|98   |107|cellulitis |Diagnosis|
|110  |119|pneumonias |Diagnosis|
+-----+---+-----------+---------+

```
```python
# Define a pipeline with 2 different NER models with a ChunkMergeApproach at the end
data = spark.createDataFrame(["A 63-year-old man presents to the hospital ..."]).toDF("text")
pipeline = Pipeline(stages=[
 DocumentAssembler(].setInputCol("text").setOutputCol("document"),
 SentenceDetector().setInputCols(["document"]).setOutputCol("sentence"),
 Tokenizer().setInputCols(["sentence"]).setOutputCol("token"),
  WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models").setOutputCol("embs"),
  MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embs"]).setOutputCol("jsl_ner"),
 NerConverter().setInputCols(["sentence", "token", "jsl_ner"]).setOutputCol("jsl_ner_chunk"),
  MedicalNerModel.pretrained("ner_bionlp", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embs"]).setOutputCol("bionlp_ner"),
 NerConverter().setInputCols(["sentence", "token", "bionlp_ner"]) \
    .setOutputCol("bionlp_ner_chunk"),
 ChunkMergeApproach().setInputCols(["jsl_ner_chunk", "bionlp_ner_chunk"]).setOutputCol("merged_chunk")
))

# Show results
result = pipeline.fit(data).transform(data).cache()
result.selectExpr("explode(merged_chunk) as a") \
  .selectExpr("a.begin","a.end","a.result as chunk","a.metadata.entity as entity") \
  .show(5, False)
+-----+---+-----------+---------+
|begin|end|chunk      |entity   |
+-----+---+-----------+---------+
|5    |15 |63-year-old|Age      |
|17   |19 |man        |Gender   |
|64   |72 |recurrent  |Modifier |
|98   |107|cellulitis |Diagnosis|
|110  |119|pneumonias |Diagnosis|
+-----+---+-----------+---------+

```
</div>

</details>

</div>

<div class="h3-box" markdown="1">
### IOBTagger

<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/ner/IOBTagger.html">Transformer scaladocs</a>

Merges token tags and NER labels from chunks in the specified format. For example output columns as inputs from NerConverter and Tokenizer can be used to merge.

**Input types:** `TOKEN, CHUNK`

**Output type:** `NAMED_ENTITY`

<details>
<summary><b>Show Example </b></summary>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}

```scala
// Pipeline stages are defined where NER is done. NER is converted to chunks.
val data = Seq(("A 63-year-old man presents to the hospital ...")).toDF("text")
val docAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
val sentenceDetector = new SentenceDetector().setInputCols("document").setOutputCol("sentence")
val tokenizer = new Tokenizer().setInputCols("sentence").setOutputCol("token")
val embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models").setOutputCol("embs")
val nerModel = MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models").setInputCols("sentence", "token", "embs").setOutputCol("ner")
val nerConverter = new NerConverter().setInputCols("sentence", "token", "ner").setOutputCol("ner_chunk")

// Define the IOB tagger, which needs tokens and chunks as input. Show results.
val iobTagger = new IOBTagger().setInputCols("token", "ner_chunk").setOutputCol("ner_label")
val pipeline = new Pipeline().setStages(Array(docAssembler, sentenceDetector, tokenizer, embeddings, nerModel, nerConverter, iobTagger))

result.selectExpr("explode(ner_label) as a")
  .selectExpr("a.begin","a.end","a.result as chunk","a.metadata.word as word")
  .where("chunk!='O'").show(5, false)

+-----+---+-----------+-----------+
|begin|end|chunk      |word       |
+-----+---+-----------+-----------+
|5    |15 |B-Age      |63-year-old|
|17   |19 |B-Gender   |man        |
|64   |72 |B-Modifier |recurrent  |
|98   |107|B-Diagnosis|cellulitis |
|110  |119|B-Diagnosis|pneumonias |
+-----+---+-----------+-----------+

```
```python
# Pipeline stages are defined where NER is done. NER is converted to chunks.
data = spark.createDataFrame(["A 63-year-old man presents to the hospital ..."]).toDF("text")
docAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
sentenceDetector = SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
tokenizer = Tokenizer().setInputCols(["sentence"]).setOutputCol("token")
embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models").setOutputCol("embs")
nerModel = MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models").setInputCols(["sentence", "token", "embs"]).setOutputCol("ner")
nerConverter = NerConverter().setInputCols(["sentence", "token", "ner"]).setOutputCol("ner_chunk")

# Define the IOB tagger, which needs tokens and chunks as input. Show results.
iobTagger = IOBTagger().setInputCols(["token", "ner_chunk"]).setOutputCol("ner_label")
pipeline = Pipeline(stages=[docAssembler, sentenceDetector, tokenizer, embeddings, nerModel, nerConverter, iobTagger])

result.selectExpr("explode(ner_label) as a") \
  .selectExpr("a.begin","a.end","a.result as chunk","a.metadata.word as word") \
  .where("chunk!='O'").show(5, False)

+-----+---+-----------+-----------+
|begin|end|chunk      |word       |
+-----+---+-----------+-----------+
|5    |15 |B-Age      |63-year-old|
|17   |19 |B-Gender   |man        |
|64   |72 |B-Modifier |recurrent  |
|98   |107|B-Diagnosis|cellulitis |
|110  |119|B-Diagnosis|pneumonias |
+-----+---+-----------+-----------+

```
</div>

</details>

</div>

<div class="h3-box" markdown="1">
### NerConverterInternal

<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/ner/NerConverterInternal.html">Transformer scaladocs</a>

Converts a IOB or IOB2 representation of NER to a user-friendly one, by associating the tokens of recognized entities and their label. Chunks with no associated entity (tagged "O") are filtered. See also Inside–outside–beginning (tagging) for more information.

**Input types:** `DOCUMENT, TOKEN, NAMED_ENTITY`

**Output type:** `CHUNK`

<details>
<summary><b>Show Example </b></summary>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}

```scala
// The output of a MedicalNerModel follows the Annotator schema and looks like this after the transformation.
result.selectExpr("explode(ner_result)").show(5, false)
+--------------------------------------------------------------------------+
|col                                                                       |
+--------------------------------------------------------------------------+
|{named_entity, 3, 3, O, {word -> A, confidence -> 0.994}, []}             |
|{named_entity, 5, 15, B-Age, {word -> 63-year-old, confidence -> 1.0}, []}|
|{named_entity, 17, 19, B-Gender, {word -> man, confidence -> 0.9858}, []} |
|{named_entity, 21, 28, O, {word -> presents, confidence -> 0.9952}, []}   |
|{named_entity, 30, 31, O, {word -> to, confidence -> 0.7063}, []}         |
+--------------------------------------------------------------------------+

// After the converter is used:
result.selectExpr("explode(ner_converter_result)").show(5, false)
+-----------------------------------------------------------------------------------+
|col                                                                                |
+-----------------------------------------------------------------------------------+
|{chunk, 5, 15, 63-year-old, {entity -> Age, sentence -> 0, chunk -> 0}, []}        |
|{chunk, 17, 19, man, {entity -> Gender, sentence -> 0, chunk -> 1}, []}            |
|{chunk, 64, 72, recurrent, {entity -> Modifier, sentence -> 0, chunk -> 2}, []}    |
|{chunk, 98, 107, cellulitis, {entity -> Diagnosis, sentence -> 0, chunk -> 3}, []} |
|{chunk, 110, 119, pneumonias, {entity -> Diagnosis, sentence -> 0, chunk -> 4}, []}|
+-----------------------------------------------------------------------------------+

```
```python
# The output of a MedicalNerModel follows the Annotator schema and looks like this after the transformation.
result.selectExpr("explode(ner_result)").show(5, False)
+--------------------------------------------------------------------------+
|col                                                                       |
+--------------------------------------------------------------------------+
|{named_entity, 3, 3, O, {word -> A, confidence -> 0.994}, []}             |
|{named_entity, 5, 15, B-Age, {word -> 63-year-old, confidence -> 1.0}, []}|
|{named_entity, 17, 19, B-Gender, {word -> man, confidence -> 0.9858}, []} |
|{named_entity, 21, 28, O, {word -> presents, confidence -> 0.9952}, []}   |
|{named_entity, 30, 31, O, {word -> to, confidence -> 0.7063}, []}         |
+--------------------------------------------------------------------------+

# After the converter is used:
result.selectExpr("explode(ner_converter_result)").show(5, False)
+-----------------------------------------------------------------------------------+
|col                                                                                |
+-----------------------------------------------------------------------------------+
|{chunk, 5, 15, 63-year-old, {entity -> Age, sentence -> 0, chunk -> 0}, []}        |
|{chunk, 17, 19, man, {entity -> Gender, sentence -> 0, chunk -> 1}, []}            |
|{chunk, 64, 72, recurrent, {entity -> Modifier, sentence -> 0, chunk -> 2}, []}    |
|{chunk, 98, 107, cellulitis, {entity -> Diagnosis, sentence -> 0, chunk -> 3}, []} |
|{chunk, 110, 119, pneumonias, {entity -> Diagnosis, sentence -> 0, chunk -> 4}, []}|
+-----------------------------------------------------------------------------------+

```
</div>

</details>

</div>

<div class="h3-box" markdown="1">
### NerDisambiguator

<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/disambiguation/NerDisambiguator.html">Transformer scaladocs</a>

Links words of interest, such as names of persons, locations and companies, from an input text document to a corresponding unique entity in a target Knowledge Base (KB). Words of interest are called Named Entities (NEs), mentions, or surface forms. The model needs extracted CHUNKS and SENTENCE_EMBEDDINGS type input from e.g. SentenceEmbeddings and NerConverter.

**Input types:** `CHUNK, SENTENCE_EMBEDDINGS`

**Output type:** `DISAMBIGUATION`

<details>
<summary><b>Show Example </b></summary>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}

```scala
// Extracting Person identities
// First define pipeline stages that extract entities and embeddings. Entities are filtered for PER type entities.
val data = Seq("The show also had a contestant named Donald Trump who later defeated Christina Aguilera ...")
  .toDF("text")
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")
val sentenceDetector = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")
val tokenizer = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")
val word_embeddings = WordEmbeddingsModel.pretrained()
  .setInputCols("sentence", "token")
  .setOutputCol("embeddings")
val sentence_embeddings = new SentenceEmbeddings()
  .setInputCols("sentence","embeddings")
  .setOutputCol("sentence_embeddings")
val ner_model = NerDLModel.pretrained()
  .setInputCols("sentence", "token", "embeddings")
  .setOutputCol("ner")
val ner_converter = new NerConverter()
  .setInputCols("sentence", "token", "ner")
  .setOutputCol("ner_chunk")
  .setWhiteList("PER")

// Then the extracted entities can be disambiguated.
val disambiguator = new NerDisambiguator()
  .setS3KnowledgeBaseName("i-per")
  .setInputCols("ner_chunk", "sentence_embeddings")
  .setOutputCol("disambiguation")
  .setNumFirstChars(5)

val nlpPipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  word_embeddings,
  sentence_embeddings,
  ner_model,
  ner_converter,
  disambiguator))

val model = nlpPipeline.fit(data)
val result = model.transform(data)

// Show results
result.selectExpr("explode(disambiguation)")
  .selectExpr("col.metadata.chunk as chunk", "col.result as result").show(5, false)
+------------------+------------------------------------------------------------------------------------------------------------------------+
|chunk             |result                                                                                                                  |
+------------------+------------------------------------------------------------------------------------------------------------------------+
|Donald Trump      |http://en.wikipedia.org/?curid=4848272, http://en.wikipedia.org/?curid=31698421, http://en.wikipedia.org/?curid=55907961|
|Christina Aguilera|http://en.wikipedia.org/?curid=144171, http://en.wikipedia.org/?curid=6636454                                           |
+------------------+------------------------------------------------------------------------------------------------------------------------+

```
```python
# Extracting Person identities
# First define pipeline stages that extract entities and embeddings. Entities are filtered for PER type entities.
data = spark.createDataframe(["The show also had a contestant named Donald Trump who later defeated Christina Aguilera ..."])
  .toDF("text")
documentAssembler = DocumentAssembler() \
  .setInputCol("text") \
  .setOutputCol("document")
sentenceDetector = SentenceDetector() \
  .setInputCols(["document"]) \
  .setOutputCol("sentence")
tokenizer = Tokenizer() \
  .setInputCols(["sentence"]) \
  .setOutputCol("token")
word_embeddings = WordEmbeddingsModel.pretrained() \
  .setInputCols(["sentence", "token"]) \
  .setOutputCol("embeddings")
sentence_embeddings = SentenceEmbeddings() \
  .setInputCols(["sentence","embeddings"]) \
  .setOutputCol("sentence_embeddings")
ner_model = NerDLModel.pretrained() \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")
ner_converter = NerConverter() \
  .setInputCols(["sentence", "token", "ner"]) \
  .setOutputCol("ner_chunk") \
  .setWhiteList("PER")

# Then the extracted entities can be disambiguated.
disambiguator = NerDisambiguator() \
  .setS3KnowledgeBaseName("i-per") \
  .setInputCols(["ner_chunk", "sentence_embeddings"]) \
  .setOutputCol("disambiguation") \
  .setNumFirstChars(5)

nlpPipeline = Pipeline(stages=[
  documentAssembler,
  sentenceDetector,
  tokenizer,
  word_embeddings,
  sentence_embeddings,
  ner_model,
  ner_converter,
  disambiguator])

model = nlpPipeline.fit(data)
result = model.transform(data)

# Show results
result.selectExpr("explode(disambiguation)")
  .selectExpr("col.metadata.chunk as chunk", "col.result as result").show(5, False)
+------------------+------------------------------------------------------------------------------------------------------------------------+
|chunk             |result                                                                                                                  |
+------------------+------------------------------------------------------------------------------------------------------------------------+
|Donald Trump      |http://en.wikipedia.org/?curid=4848272, http://en.wikipedia.org/?curid=31698421, http://en.wikipedia.org/?curid=55907961|
|Christina Aguilera|http://en.wikipedia.org/?curid=144171, http://en.wikipedia.org/?curid=6636454                                           |
+------------------+------------------------------------------------------------------------------------------------------------------------+

```
</div>

</details>

</div>

<div class="h3-box" markdown="1">

### SentenceEntityResolver
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/resolution/SentenceEntityResolverApproach.html">Estimator scaladocs</a> |
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/resolution/SentenceEntityResolverModel.html">Transformer scaladocs</a>

Assigns a standard code (ICD10 CM, PCS, ICDO; CPT) to sentence embeddings pooled over chunks from TextMatchers or the NER Models.
This annotator is particularly handy when workING with BertSentenceEmbeddings from the upstream chunks.

**Input types:** `SENTENCE_EMBEDDINGS`

**Output type:** `ENTITY`

<details>
<summary><b>Show Example </b></summary>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}

```scala
// Training a SNOMED resolution model using BERT sentence embeddings
// Define pre-processing pipeline for training data. It needs consists of columns for the normalized training data and their labels.
val documentAssembler = new DocumentAssembler()
  .setInputCol("normalized_text")
  .setOutputCol("document")
val bertEmbeddings = BertSentenceEmbeddings.pretrained("sent_biobert_pubmed_base_cased")
  .setInputCols("sentence")
  .setOutputCol("bert_embeddings")
val snomedTrainingPipeline = new Pipeline().setStages(Array(
  documentAssembler,
  bertEmbeddings
))
val snomedTrainingModel = snomedTrainingPipeline.fit(data)
val snomedData = snomedTrainingModel.transform(data).cache()

// Then the Resolver can be trained with
val bertExtractor = new SentenceEntityResolverApproach()
  .setNeighbours(25)
  .setThreshold(1000)
  .setInputCols("bert_embeddings")
  .setNormalizedCol("normalized_text")
  .setLabelCol("label")
  .setOutputCol("snomed_code")
  .setDistanceFunction("EUCLIDIAN")
  .setCaseSensitive(false)

val snomedModel = bertExtractor.fit(snomedData)

```
```python
# Training a SNOMED resolution model using BERT sentence embeddings
# Define pre-processing pipeline for training data. It needs consists of columns for the normalized training data and their labels.
documentAssembler = DocumentAssembler() \
  .setInputCol("normalized_text") \
  .setOutputCol("document")
bertEmbeddings = BertSentenceEmbeddings.pretrained("sent_biobert_pubmed_base_cased") \
  .setInputCols(["sentence"]) \
  .setOutputCol("bert_embeddings")
snomedTrainingPipeline = Pipeline(stages=[
  documentAssembler,
  bertEmbeddings
])
snomedTrainingModel = snomedTrainingPipeline.fit(data)
snomedData = snomedTrainingModel.transform(data).cache()

# Then the Resolver can be trained with
bertExtractor = SentenceEntityResolverApproach() \
  .setNeighbours(25) \
  .setThreshold(1000) \
  .setInputCols(["bert_embeddings"]) \
  .setNormalizedCol("normalized_text") \
  .setLabelCol("label") \
  .setOutputCol("snomed_code") \
  .setDistanceFunction("EUCLIDIAN") \
  .setCaseSensitive(False)

snomedModel = bertExtractor.fit(snomedData)

```

</div>

</details>

</div>

<div class="h3-box" markdown="1">

### DocumentLogRegClassifier
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/classification/DocumentLogRegClassifierApproach.html">Estimator scaladocs</a> |
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/classification/DocumentLogRegClassifierModel.html">Transformer scaladocs</a>

A convenient TFIDF-LogReg classifier that accepts "token" input type and outputs "selector"; an input type mainly used in RecursivePipelineModels

**Input types:** `TOKEN`

**Output type:** `CATEGORY`

**Example:**

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
logregClassifier = DocumentLogRegClassifierApproach() \
    .setInputCols("chunk_token") \
    .setOutputCol("category") \
    .setLabelCol("label_col") \
    .setMaxIter(10) \
    .setTol(1e-6) \
    .setFitIntercept(True)
```
```scala
val logregClassifier = new DocumentLogRegClassifierApproach()
    .setInputCols("chunk_token")
    .setOutputCol("category")
    .setLabelCol("label_col")
    .setMaxIter(10)
    .setTol(1e-6)
    .setFitIntercept(true)
```

</div></div><div class="h3-box" markdown="1">

### DeIdentificator
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/deid/DeIdentification.html">Estimator scaladocs</a> |
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/deid/DeIdentificationModel.html">Transformer scaladocs</a>

Identifies potential pieces of content with personal information about patients and remove them by replacing with semantic tags.

**Input types:** `DOCUMENT, TOKEN, CHUNK`

**Output type:** `DOCUMENT`

**Example:**

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
deid = DeIdentificationApproach() \
      .setInputCols("sentence", "token", "ner_chunk") \
      .setOutputCol("deid_sentence") \
      .setRegexPatternsDictionary("src/test/resources/de-identification/dic_regex_patterns_main_categories.txt") \
      .setMode("mask") \
      .setDateTag("DATE") \
      .setObfuscateDate(False) \
      .setDays(5) \
      .setDateToYear(False) \
      .setMinYear(1900) \
      .setDateFormats(["MM-dd-yyyy","MM-dd-yy"]) \
      .setConsistentObfuscation(True) \
      .setSameEntityThreshold(0.9)
```
```scala
val deid = new DeIdentificationApproach()
      .setInputCols("sentence", "token", "ner_chunk")
      .setOutputCol("deid_sentence")
      .setRegexPatternsDictionary("src/test/resources/de-identification/dic_regex_patterns_main_categories.txt") \
      .setMode("mask")
      .setDateTag("DATE")
      .setObfuscateDate(false)
      .setDays(5)
      .setDateToYear(false)
      .setMinYear(1900)
      .setDateFormats(Seq("MM-dd-yyyy","MM-dd-yy"))
      .setConsistentObfuscation(true)
      .setSameEntityThreshold(0.9)
```

</div></div><div class="h3-box" markdown="1">

### Contextual Parser
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/context/ContextualParserApproach.html">Estimator scaladocs</a> |
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/context/ContextualParserModel.html">Transformer scaladocs</a>

This annotator provides Regex + Contextual Matching, based on a JSON file.

**Output type:** `DOCUMENT, TOKEN`

**Input types:** `CHUNK`

**JSON format:**
```
{
  "entity": "Stage",
  "ruleScope": "sentence",
  "regex": "[cpyrau]?[T][0-9X?][a-z^cpyrau]*",
  "matchScope": "token"
}
```

- setJsonPath() -> Path to json file with rules
- setCaseSensitive() -> optional: Whether to use case sensitive when matching values, default is false
- setPrefixAndSuffixMatch() -> optional: Whether to force both before AND after the regex match to annotate the hit
- setContextMatch() -> optional: Whether to include prior and next context to annotate the hit
- setUpdateTokenizer() -> optional: Whether to update tokenizer from pipeline when detecting multiple words on dictionary values
- setDictionary() -> optional: Path to dictionary file in tsv or csv format

**Example:**

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
contextual_parser = ContextualParserApproach() \
        .setInputCols(["sentence", "token"]) \
        .setOutputCol("entity_stage") \
        .setJsonPath("data/Stage.json")
```
```scala
val contextualParser = new ContextualParserApproach()
        .setInputCols(Array("sentence", "token"))
        .setOutputCol("entity_stage")
        .setJsonPath("data/Stage.json")
```

</div></div><div class="h3-box" markdown="1">

### RelationExtraction
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/re/RelationExtractionApproach.html">Estimator scaladocs</a> |
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/re/RelationExtractionModel.html">Transformer scaladocs</a>

Extracts and classifier instances of relations between named entities.

**Input types:** `WORD_EMBEDDINGS, POS, CHUNK, DEPENDENCY`

**Output type:** `CATEGORY`

**Example:**

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
reApproach = sparknlp_jsl.annotator.RelationExtractionApproach()\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setLabelColumn("target_rel")\
    .setEpochsNumber(300)\
    .setBatchSize(200)\
    .setLearningRate(0.001)\
    .setModelFile("RE.in1200D.out20.pb")\
    .setFixImbalance(True)\
    .setValidationSplit(0.05)\
    .setFromEntity("from_begin", "from_end", "from_label")\
    .setToEntity("to_begin", "to_end", "to_label")
```

```scala
val reApproach = new RelationExtractionApproach()
  .setInputCols(Array("embeddings", "pos_tags", "ner_chunks", "dependencies"))
  .setOutputCol("relations")
  .setLabelColumn("target_rel")
  .setEpochsNumber(300)
  .setBatchSize(200)
  .setlearningRate(0.001f)
  .setModelFile("RE.in1200D.out20.pb")
  .setFixImbalance(true)
  .setValidationSplit(0.05f)
  .setFromEntity("from_begin", "from_end", "from_label")
  .setToEntity("to_begin", "to_end", "to_label")

```

</div>