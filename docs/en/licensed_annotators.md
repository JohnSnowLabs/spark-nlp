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
Check out the [Spark NLP Annotators page](https://nlp.johnsnowlabs.com/docs/en/annotators) for more information.

</div>

<div class="h3-box" markdown="1">

### AssertionDL
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/assertion/dl/AssertionDLApproach.html">Approach scaladocs</a> |
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/assertion/dl/AssertionDLModel.html">Model scaladocs</a>

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
  ["Patient with severe fever and sore throat"],
  ["Patient shows no stomach pain"],
  ["She was maintained on an epidural and PCA for pain control."]]).toDF("text")
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

<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/chunker/AssertionFilterer.html">Model scaladocs</a>

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
  .setWhiteList(["present"])

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

### AssertionLogReg
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/assertion/logreg/AssertionLogRegApproach.html">Approach scaladocs</a> |
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/assertion/logreg/AssertionLogRegModel.html">Model scaladocs</a>

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

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
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

### Chunk2Token

<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/Chunk2Token.html">API scaladocs</a>

A feature transformer that converts the input array of strings (annotatorType CHUNK) into an array of chunk-based tokens (annotatorType TOKEN). When the input is empty, an empty array is returned. This Annotator is specially convenient when using NGramGenerator annotations as inputs to WordEmbeddingsModels

**Input types:** `CHUNK`

**Output type:** `TOKEN`

<details>
<summary><b>Show Example</b></summary>

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}
```scala
// Define a pipeline for generating n-grams
val data = Seq(("A 63-year-old man presents to the hospital ...")).toDF("text")
val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")
val sentenceDetector = new SentenceDetector().setInputCols("document").setOutputCol("sentence")
val token = new Tokenizer().setInputCols("sentence").setOutputCol("token")
val ngrammer = new NGramGenerator()
 .setN(2)
 .setEnableCumulative(false)
 .setInputCols("token")
 .setOutputCol("ngrams")
 .setDelimiter("_")

// Stage to convert n-gram CHUNKS to TOKEN type
val chunk2Token = new Chunk2Token().setInputCols("ngrams").setOutputCol("ngram_tokens")
val trainingPipeline = new Pipeline().setStages(Array(document, sentenceDetector, token, ngrammer, chunk2Token)).fit(data)

val result = trainingPipeline.transform(data).cache()
result.selectExpr("explode(ngram_tokens)").show(5, false)
  +----------------------------------------------------------------+
  |col                                                             |
  +----------------------------------------------------------------+
  |{token, 3, 15, A_63-year-old, {sentence -> 0, chunk -> 0}, []}  |
  |{token, 5, 19, 63-year-old_man, {sentence -> 0, chunk -> 1}, []}|
  |{token, 17, 28, man_presents, {sentence -> 0, chunk -> 2}, []}  |
  |{token, 21, 31, presents_to, {sentence -> 0, chunk -> 3}, []}   |
  |{token, 30, 35, to_the, {sentence -> 0, chunk -> 4}, []}        |
  +----------------------------------------------------------------+

```
```python
# Define a pipeline for generating n-grams
data = spark.createDataFrame([["A 63-year-old man presents to the hospital ..."]]).toDF("text")
document = DocumentAssembler().setInputCol("text").setOutputCol("document")
sentenceDetector = SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
token = Tokenizer().setInputCols(["sentence"]).setOutputCol("token")
ngrammer = NGramGenerator() \
 .setN(2) \
 .setEnableCumulative(False) \
 .setInputCols(["token"]) \
 .setOutputCol("ngrams") \
 .setDelimiter("_")

# Stage to convert n-gram CHUNKS to TOKEN type
chunk2Token = Chunk2Token().setInputCols(["ngrams"]).setOutputCol("ngram_tokens")
trainingPipeline = Pipeline(stages=[document, sentenceDetector, token, ngrammer, chunk2Token]).fit(data)

result = trainingPipeline.transform(data).cache()
result.selectExpr("explode(ngram_tokens)").show(5, False)
  +----------------------------------------------------------------+
  |col                                                             |
  +----------------------------------------------------------------+
  |{token, 3, 15, A_63-year-old, {sentence -> 0, chunk -> 0}, []}  |
  |{token, 5, 19, 63-year-old_man, {sentence -> 0, chunk -> 1}, []}|
  |{token, 17, 28, man_presents, {sentence -> 0, chunk -> 2}, []}  |
  |{token, 21, 31, presents_to, {sentence -> 0, chunk -> 3}, []}   |
  |{token, 30, 35, to_the, {sentence -> 0, chunk -> 4}, []}        |
  +----------------------------------------------------------------+

```
</div>

</details>

</div>

<div class="h3-box" markdown="1">

### ChunkEntityResolver
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/resolution/ChunkEntityResolverApproach.html">Approach scaladocs</a> |
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/resolution/ChunkEntityResolverModel.html">Model scaladocs</a>

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

token = Tokenizer() \
    .setInputCols(["document"]) \
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

<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/chunker/ChunkFilterer.html">Model scaladocs</a>

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

val result = pipeline.fit(data).transform(data)
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
data = spark.createDataFrame([["Has a past history of gastroenteritis and stomach pain, however patient ..."]]).toDF("text")
docAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
sentenceDetector = SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
tokenizer = Tokenizer().setInputCols(["sentence"]).setOutputCol("token")

posTagger = PerceptronModel.pretrained() \
  .setInputCols(["sentence", "token"]) \
  .setOutputCol("pos")

chunker = Chunker() \
  .setInputCols(["pos", "sentence"]) \
  .setOutputCol("chunk") \
  .setRegexParsers(["(<NN>)+"])

# Then the chunks can be filtered via a white list. Here only terms with "gastroenteritis" remain.
chunkerFilter = ChunkFilterer() \
  .setInputCols(["sentence","chunk"]) \
  .setOutputCol("filtered") \
  .setCriteria("isin") \
  .setWhiteList(["gastroenteritis"])

pipeline = Pipeline(stages=[
  docAssembler,
  sentenceDetector,
  tokenizer,
  posTagger,
  chunker,
  chunkerFilter])

result = pipeline.fit(data).transform(data)
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

<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/merge/ChunkMergeApproach.html">Approach scaladocs</a> | <a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/merge/ChunkMergeModel.html">Model scaladocs</a>

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
data = spark.createDataFrame([["A 63-year-old man presents to the hospital ..."]]).toDF("text")
pipeline = Pipeline(stages=[
 DocumentAssembler().setInputCol("text").setOutputCol("document"),
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
])

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

<div class="h3-box" markdown="1">

### ContextualParser

<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/context/ContextualParserApproach.html">Approach scaladocs</a> | <a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/context/ContextualParserModel.html">Model scaladocs</a>

Creates a model, that extracts entity from a document based on user defined rules. Rule matching is based on a RegexMatcher defined in a JSON file. It is set through the parameter setJsonPath(). In this JSON file, regex is defined that you want to match along with the information that will output on metadata field. Additionally, a dictionary can be provided with `setDictionary` to map extracted entities to a unified representation. The first column of the dictionary file should be the representation with following columns the possible matches.

An example JSON file `regex_token.json` can look like this:
```json
{
   "entity": "Stage",
   "ruleScope": "sentence",
   "regex": "[cpyrau]?[T][0-9X?][a-z^cpyrau]*",
   "matchScope": "token"
 }
```
Which extracts the stage code on a sentence level.

**Input types:** `DOCUMENT, TOKEN`

**Output type:** `CHUNK`

<details>
<summary><b>Show Example</b></summary>

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}
```scala
// Pipeline could then be defined like this
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

// Define the parser (json file needs to be provided)
val data = Seq("A patient has liver metastases pT1bN0M0 and the T5 primary site may be colon or... ").toDF("text")
val contextualParser = new ContextualParserApproach()
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("entity")
  .setJsonPath("/path/to/regex_token.json")
  .setCaseSensitive(true)
  .setContextMatch(false)
val pipeline = new Pipeline().setStages(Array(
    documentAssembler,
    sentenceDetector,
    tokenizer,
    contextualParser
  ))

val result = pipeline.fit(data).transform(data)

// Show Results
result.selectExpr("explode(entity)").show(5, truncate=false)
+-------------------------------------------------------------------------------------------------------------------------+
|col                                                                                                                      |
+-------------------------------------------------------------------------------------------------------------------------+
|{chunk, 32, 39, pT1bN0M0, {field -> Stage, normalized -> , confidenceValue -> 0.13, hits -> regex, sentence -> 0}, []}   |
|{chunk, 49, 50, T5, {field -> Stage, normalized -> , confidenceValue -> 0.13, hits -> regex, sentence -> 0}, []}         |
|{chunk, 148, 156, cT4bcN2M1, {field -> Stage, normalized -> , confidenceValue -> 0.13, hits -> regex, sentence -> 1}, []}|
|{chunk, 189, 194, T?N3M1, {field -> Stage, normalized -> , confidenceValue -> 0.13, hits -> regex, sentence -> 2}, []}   |
|{chunk, 316, 323, pT1bN0M0, {field -> Stage, normalized -> , confidenceValue -> 0.13, hits -> regex, sentence -> 3}, []} |
+-------------------------------------------------------------------------------------------------------------------------+

```
```python
# Pipeline could then be defined like this
documentAssembler = DocumentAssembler() \
  .setInputCol("text") \
  .setOutputCol("document")

sentenceDetector = SentenceDetector() \
  .setInputCols(["document"]) \
  .setOutputCol("sentence")

tokenizer = Tokenizer() \
  .setInputCols(["sentence"]) \
  .setOutputCol("token")

# Define the parser (json file needs to be provided)
data = spark.createDataFrame([["A patient has liver metastases pT1bN0M0 and the T5 primary site may be colon or... "]]).toDF("text")

contextualParser = ContextualParserApproach() \
  .setInputCols(["sentence", "token"]) \
  .setOutputCol("entity") \
  .setJsonPath("/path/to/regex_token.json") \
  .setCaseSensitive(True) \
  .setContextMatch(False)

pipeline = Pipeline(stages=[
    documentAssembler,
    sentenceDetector,
    tokenizer,
    contextualParser
  ])

result = pipeline.fit(data).transform(data)

# Show Results
result.selectExpr("explode(entity)").show(5, truncate=False)
+-------------------------------------------------------------------------------------------------------------------------+
|col                                                                                                                      |
+-------------------------------------------------------------------------------------------------------------------------+
|{chunk, 32, 39, pT1bN0M0, {field -> Stage, normalized -> , confidenceValue -> 0.13, hits -> regex, sentence -> 0}, []}   |
|{chunk, 49, 50, T5, {field -> Stage, normalized -> , confidenceValue -> 0.13, hits -> regex, sentence -> 0}, []}         |
|{chunk, 148, 156, cT4bcN2M1, {field -> Stage, normalized -> , confidenceValue -> 0.13, hits -> regex, sentence -> 1}, []}|
|{chunk, 189, 194, T?N3M1, {field -> Stage, normalized -> , confidenceValue -> 0.13, hits -> regex, sentence -> 2}, []}   |
|{chunk, 316, 323, pT1bN0M0, {field -> Stage, normalized -> , confidenceValue -> 0.13, hits -> regex, sentence -> 3}, []} |
+-------------------------------------------------------------------------------------------------------------------------+

```
</div>

</details>

</div>

<div class="h3-box" markdown="1">

### DeIdentificator
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/deid/DeIdentification.html">Approach scaladocs</a> |
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/deid/DeIdentificationModel.html">Model scaladocs</a>

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

</div></div>

<div class="h3-box" markdown="1">

### DocumentLogRegClassifier
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/classification/DocumentLogRegClassifierApproach.html">Approach scaladocs</a> |
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/classification/DocumentLogRegClassifierModel.html">Model scaladocs</a>

A convenient TFIDF-LogReg classifier that accepts "token" input type and outputs "selector"; an input type mainly used in RecursivePipelineModels

**Input types:** `TOKEN`

**Output type:** `CATEGORY`

<details>
<summary><b>Show Example</b></summary>

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}
```scala
// Define pipeline stages to prepare the data
val document_assembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val normalizer = new Normalizer()
  .setInputCols("token")
  .setOutputCol("normalized")

val stopwords_cleaner = new StopWordsCleaner()
  .setInputCols("normalized")
  .setOutputCol("cleanTokens")
  .setCaseSensitive(false)

val stemmer = new Stemmer()
  .setInputCols("cleanTokens")
  .setOutputCol("stem")

// Define the document classifier and fit training data to it
val logreg = new DocumentLogRegClassifierApproach()
  .setInputCols("stem")
  .setLabelCol("category")
  .setOutputCol("prediction")

val pipeline = new Pipeline().setStages(Array(
  document_assembler,
  tokenizer,
  normalizer,
  stopwords_cleaner,
  stemmer,
  logreg
))

val model = pipeline.fit(trainingData)

```
```python
# Define pipeline stages to prepare the data
document_assembler = DocumentAssembler() \
  .setInputCol("text") \
  .setOutputCol("document")

tokenizer = Tokenizer() \
  .setInputCols(["document"]) \
  .setOutputCol("token")

normalizer = Normalizer() \
  .setInputCols(["token"]) \
  .setOutputCol("normalized")

stopwords_cleaner = StopWordsCleaner() \
  .setInputCols(["normalized"]) \
  .setOutputCol("cleanTokens") \
  .setCaseSensitive(False)

stemmer = Stemmer() \
  .setInputCols(["cleanTokens"]) \
  .setOutputCol("stem")

# Define the document classifier and fit training data to it
logreg = DocumentLogRegClassifierApproach() \
  .setInputCols(["stem"]) \
  .setLabelCol("category") \
  .setOutputCol("prediction")

pipeline = Pipeline(stages=[
  document_assembler,
  tokenizer,
  normalizer,
  stopwords_cleaner,
  stemmer,
  logreg
])

model = pipeline.fit(trainingData)

```
</div>

</details>

</div>

<div class="h3-box" markdown="1">

### DrugNormalizer

<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/DrugNormalizer.html">API scaladocs</a>

Annotator which normalizes raw text from clinical documents, e.g. scraped web pages or xml documents, from document type columns into Sentence. Removes all dirty characters from text following one or more input regex patterns. Can apply non wanted character removal which a specific policy. Can apply lower case normalization. See Spark NLP Workshop for more examples of usage.

**Input types:** `DOCUMENT`

**Output type:** `DOCUMENT`

<details>
<summary><b>Show Example </b></summary>


<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}
```scala
val data = Seq(
  ("Sodium Chloride/Potassium Chloride 13bag"),
  ("interferon alfa-2b 10 million unit ( 1 ml ) injec"),
  ("aspirin 10 meq/ 5 ml oral sol")
).toDF("text")
val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")
val drugNormalizer = new DrugNormalizer().setInputCols("document").setOutputCol("document_normalized")

val trainingPipeline = new Pipeline().setStages(Array(document, drugNormalizer))
val result = trainingPipeline.fit(data).transform(data)

result.selectExpr("explode(document_normalized.result) as normalized_text").show(false)
+----------------------------------------------------+
|normalized_text                                     |
+----------------------------------------------------+
|Sodium Chloride / Potassium Chloride 13 bag         |
|interferon alfa - 2b 10000000 unt ( 1 ml ) injection|
|aspirin 2 meq/ml oral solution                      |
+----------------------------------------------------+

```
```python
data = spark.createDataFrame([
  ["Sodium Chloride/Potassium Chloride 13bag"],
  ["interferon alfa-2b 10 million unit ( 1 ml ) injec"],
  ["aspirin 10 meq/ 5 ml oral sol"]
]).toDF("text")
document = DocumentAssembler().setInputCol("text").setOutputCol("document")
drugNormalizer = DrugNormalizer().setInputCols(["document"]).setOutputCol("document_normalized")

trainingPipeline = Pipeline(stages=[document, drugNormalizer])
result = trainingPipeline.fit(data).transform(data)

result.selectExpr("explode(document_normalized.result) as normalized_text").show(truncate=False)
+----------------------------------------------------+
|normalized_text                                     |
+----------------------------------------------------+
|Sodium Chloride / Potassium Chloride 13 bag         |
|interferon alfa - 2b 10000000 unt ( 1 ml ) injection|
|aspirin 2 meq/ml oral solution                      |
+----------------------------------------------------+

```
</div>

</details>

</div>

<div class="h3-box" markdown="1">

### GenericClassifierApproach

<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/generic_classifier/GenericClassifierApproach.html">Approach scaladocs</a> |
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/generic_classifier/GenericClassifierModel.html">Model scaladocs</a>

Trains a TensorFlow model for generic classification of feature vectors. It takes FEATURE_VECTOR annotations from `FeaturesAssembler` as input, classifies them and outputs CATEGORY annotations. Please see the Parameters section for required training parameters. For a more extensive example please see the Spark NLP Workshop.

**Input types:** `FEATURE_VECTOR`

**Output type:** `CATEGORY`

<details>
<summary><b>Show Example</b></summary>


<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}
```scala
val features_asm = new FeaturesAssembler()
  .setInputCols(Array("feature_1", "feature_2", "...", "feature_n"))
  .setOutputCol("features")

val gen_clf = new GenericClassifierApproach()
  .setLabelColumn("target")
  .setInputCols("features")
  .setOutputCol("prediction")
  .setModelFile("/path/to/graph_file.pb")
  .setEpochsNumber(50)
  .setBatchSize(100)
  .setFeatureScaling("zscore")
  .setlearningRate(0.001f)
  .setFixImbalance(true)
  .setOutputLogsPath("logs")
  .setValidationSplit(0.2f) // keep 20% of the data for validation purposes

val pipeline = new Pipeline().setStages(Array(
  features_asm,
  gen_clf
))

val clf_model = pipeline.fit(data)

```
```python
features_asm = FeaturesAssembler() \
  .setInputCols(["feature_1", "feature_2", "...", "feature_n"]) \
  .setOutputCol("features")

gen_clf = GenericClassifierApproach() \
  .setLabelColumn("target") \
  .setInputCols(["features"]) \
  .setOutputCol("prediction") \
  .setModelFile("/path/to/graph_file.pb") \
  .setEpochsNumber(50) \
  .setBatchSize(100) \
  .setFeatureScaling("zscore") \
  .setLearningRate(0.001) \
  .setFixImbalance(True) \
  .setOutputLogsPath("logs") \
  .setValidationSplit(0.2) # keep 20% of the data for validation purposes

pipeline = Pipeline(stages=[
  features_asm,
  gen_clf
])

clf_model = pipeline.fit(data)

```
</div>

</details>

</div>

<div class="h3-box" markdown="1">

### IOBTagger

<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/ner/IOBTagger.html">API scaladocs</a>

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
data = spark.createDataFrame([["A 63-year-old man presents to the hospital ..."]]).toDF("text")
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

### NerChunker

<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/ner/NerChunker.html">API scaladocs</a>

Extracts phrases that fits into a known pattern using the NER tags. Useful for entity groups with neighboring tokens when there is no pretrained NER model to address certain issues. A Regex needs to be provided to extract the tokens between entities.

**Input types:** `DOCUMENT, NAMED_ENTITY`

**Output type:** `CHUNK`

<details>
<summary><b>Show Example</b></summary>


<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}
```scala
// Defining pipeline stages for NER
val data= Seq("She has cystic cyst on her kidney.").toDF("text")

val documentAssembler=new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector=new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")
  .setUseAbbreviations(false)

val tokenizer=new Tokenizer()
  .setInputCols(Array("sentence"))
  .setOutputCol("token")

val embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols("sentence","token")
  .setOutputCol("embeddings")
  .setCaseSensitive(false)

val ner = MedicalNerModel.pretrained("ner_radiology", "en", "clinical/models")
  .setInputCols("sentence","token","embeddings")
  .setOutputCol("ner")
  .setIncludeConfidence(true)

// Define the NerChunker to combine to chunks
val chunker = new NerChunker()
  .setInputCols(Array("sentence","ner"))
  .setOutputCol("ner_chunk")
  .setRegexParsers(Array("<ImagingFindings>.*<BodyPart>"))

val pipeline=new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings,
  ner,
  chunker
))

val result = pipeline.fit(data).transform(data)

// Show results:
result.selectExpr("explode(arrays_zip(ner.metadata , ner.result))")
  .selectExpr("col['0'].word as word" , "col['1'] as ner").show(truncate=false)
+------+-----------------+
|word  |ner              |
+------+-----------------+
|She   |O                |
|has   |O                |
|cystic|B-ImagingFindings|
|cyst  |I-ImagingFindings|
|on    |O                |
|her   |O                |
|kidney|B-BodyPart       |
|.     |O                |
+------+-----------------+

result.select("ner_chunk.result").show(truncate=false)
+---------------------------+
|result                     |
+---------------------------+
|[cystic cyst on her kidney]|
+---------------------------+

```
```python
# Defining pipeline stages for NER
data= spark.createDataFrame([["She has cystic cyst on her kidney."]]).toDF("text")

documentAssembler= DocumentAssembler() \
  .setInputCol("text") \
  .setOutputCol("document")

sentenceDetector= SentenceDetector() \
  .setInputCols(["document"]) \
  .setOutputCol("sentence") \
  .setUseAbbreviations(False)

tokenizer= Tokenizer() \
  .setInputCols(["sentence"]) \
  .setOutputCol("token")

embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models") \
  .setInputCols(["sentence","token"]) \
  .setOutputCol("embeddings") \
  .setCaseSensitive(False)

ner = MedicalNerModel.pretrained("ner_radiology", "en", "clinical/models") \
  .setInputCols(["sentence","token","embeddings"]) \
  .setOutputCol("ner") \
  .setIncludeConfidence(True)

# Define the NerChunker to combine to chunks
chunker = NerChunker() \
  .setInputCols(["sentence","ner"]) \
  .setOutputCol("ner_chunk") \
  .setRegexParsers(["<ImagingFindings>.*<BodyPart>"])

pipeline= Pipeline(stages=[
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings,
  ner,
  chunker
])

result = pipeline.fit(data).transform(data)

# Show results:
result.selectExpr("explode(arrays_zip(ner.metadata , ner.result))") \
  .selectExpr("col['0'].word as word" , "col['1'] as ner").show(truncate=False)
+------+-----------------+
|word  |ner              |
+------+-----------------+
|She   |O                |
|has   |O                |
|cystic|B-ImagingFindings|
|cyst  |I-ImagingFindings|
|on    |O                |
|her   |O                |
|kidney|B-BodyPart       |
|.     |O                |
+------+-----------------+

result.select("ner_chunk.result").show(truncate=False)
+---------------------------+
|result                     |
+---------------------------+
|[cystic cyst on her kidney]|
+---------------------------+

```
</div>

</details>

</div>

<div class="h3-box" markdown="1">

### NerConverterInternal

<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/ner/NerConverterInternal.html">API scaladocs</a>

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

<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/disambiguation/NerDisambiguator.html">API scaladocs</a>

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
data = spark.createDataFrame([["The show also had a contestant named Donald Trump who later defeated Christina Aguilera ..."]]) \
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
  .setWhiteList(["PER"])

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
result.selectExpr("explode(disambiguation)") \
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

### RelationExtraction
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/re/RelationExtractionApproach.html">Approach scaladocs</a> |
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/re/RelationExtractionModel.html">Model scaladocs</a>

Extracts and classifies instances of relations between named entities.

**Input types:** `WORD_EMBEDDINGS, POS, CHUNK, DEPENDENCY`

**Output type:** `CATEGORY`

<details>
<summary><b>Show Example</b></summary>

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}
```scala
// Defining pipeline stages to extract entities first
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols(Array("document"))
  .setOutputCol("tokens")

val embedder = WordEmbeddingsModel
  .pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("document", "tokens"))
  .setOutputCol("embeddings")

val posTagger = PerceptronModel
  .pretrained("pos_clinical", "en", "clinical/models")
  .setInputCols(Array("document", "tokens"))
  .setOutputCol("posTags")

val nerTagger = MedicalNerModel
  .pretrained("ner_events_clinical", "en", "clinical/models")
  .setInputCols(Array("document", "tokens", "embeddings"))
  .setOutputCol("ner_tags")

val nerConverter = new NerConverter()
  .setInputCols(Array("document", "tokens", "ner_tags"))
  .setOutputCol("nerChunks")

val depencyParser = DependencyParserModel
  .pretrained("dependency_conllu", "en")
  .setInputCols(Array("document", "posTags", "tokens"))
  .setOutputCol("dependencies")

// Then define `RelationExtractionApproach` and training parameters
val re = new RelationExtractionApproach()
  .setInputCols(Array("embeddings", "posTags", "train_ner_chunks", "dependencies"))
  .setOutputCol("relations_t")
  .setLabelColumn("target_rel")
  .setEpochsNumber(300)
  .setBatchSize(200)
  .setlearningRate(0.001f)
  .setModelFile("path/to/graph_file.pb")
  .setFixImbalance(true)
  .setValidationSplit(0.05f)
  .setFromEntity("from_begin", "from_end", "from_label")
  .setToEntity("to_begin", "to_end", "to_label")

val finisher = new Finisher()
  .setInputCols(Array("relations_t"))
  .setOutputCols(Array("relations"))
  .setCleanAnnotations(false)
  .setValueSplitSymbol(",")
  .setAnnotationSplitSymbol(",")
  .setOutputAsArray(false)

// Define complete pipeline and start training
val pipeline = new Pipeline()
  .setStages(Array(
    documentAssembler,
    tokenizer,
    embedder,
    posTagger,
    nerTagger,
    nerConverter,
    depencyParser,
    re,
    finisher))

val model = pipeline.fit(trainData)

```
```python
# Defining pipeline stages to extract entities first
documentAssembler = DocumentAssembler() \
  .setInputCol("text") \
  .setOutputCol("document")

tokenizer = Tokenizer() \
  .setInputCols(["document"]) \
  .setOutputCol("tokens")

embedder = WordEmbeddingsModel \
  .pretrained("embeddings_clinical", "en", "clinical/models") \
  .setInputCols(["document", "tokens"]) \
  .setOutputCol("embeddings")

posTagger = PerceptronModel \
  .pretrained("pos_clinical", "en", "clinical/models") \
  .setInputCols(["document", "tokens"]) \
  .setOutputCol("posTags")

nerTagger = MedicalNerModel \
  .pretrained("ner_events_clinical", "en", "clinical/models") \
  .setInputCols(["document", "tokens", "embeddings"]) \
  .setOutputCol("ner_tags")

nerConverter = NerConverter() \
  .setInputCols(["document", "tokens", "ner_tags"]) \
  .setOutputCol("nerChunks")

depencyParser = DependencyParserModel \
  .pretrained("dependency_conllu", "en") \
  .setInputCols(["document", "posTags", "tokens"]) \
  .setOutputCol("dependencies")

# Then define `RelationExtractionApproach` and training parameters
re = RelationExtractionApproach() \
  .setInputCols(["embeddings", "posTags", "train_ner_chunks", "dependencies"]) \
  .setOutputCol("relations_t") \
  .setLabelColumn("target_rel") \
  .setEpochsNumber(300) \
  .setBatchSize(200) \
  .setLearningRate(0.001) \
  .setModelFile("path/to/graph_file.pb") \
  .setFixImbalance(True) \
  .setValidationSplit(0.05) \
  .setFromEntity("from_begin", "from_end", "from_label") \
  .setToEntity("to_begin", "to_end", "to_label")

finisher = Finisher() \
  .setInputCols(["relations_t"]) \
  .setOutputCols(["relations"]) \
  .setCleanAnnotations(False) \
  .setValueSplitSymbol(",") \
  .setAnnotationSplitSymbol(",") \
  .setOutputAsArray(False)

# Define complete pipeline and start training
pipeline = Pipeline(stages=[
    documentAssembler,
    tokenizer,
    embedder,
    posTagger,
    nerTagger,
    nerConverter,
    depencyParser,
    re,
    finisher])

model = pipeline.fit(trainData)

```
</div>

</details>

</div>

<div class="h3-box" markdown="1">

### RelationExtractionDL

<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/re/RelationExtractionDLModel.html">Model scaladocs</a>

Extracts and classifies instances of relations between named entities. In contrast with RelationExtractionModel, RelationExtractionDLModel is based on BERT. For pretrained models please see the Models Hub for available models.

**Input types:** `CHUNK, DOCUMENT`

**Output type:** `CATEGORY`

<details>
<summary><b>Show Example</b></summary>

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}
```scala
// Relation Extraction between body parts
// This is a continuation of the RENerChunksFilter example. See that class on how to extract the relation chunks.
// Define the extraction model
val re_ner_chunk_filter = new RENerChunksFilter()
 .setInputCols("ner_chunks", "dependencies")
 .setOutputCol("re_ner_chunks")
 .setMaxSyntacticDistance(4)
 .setRelationPairs(Array("internal_organ_or_component-direction"))

val re_model = RelationExtractionDLModel.pretrained("redl_bodypart_direction_biobert", "en", "clinical/models")
  .setPredictionThreshold(0.5f)
  .setInputCols("re_ner_chunks", "sentences")
  .setOutputCol("relations")

val trained_pipeline = new Pipeline().setStages(Array(
  documenter,
  sentencer,
  tokenizer,
  words_embedder,
  pos_tagger,
  clinical_ner_tagger,
  ner_chunker,
  dependency_parser,
  re_ner_chunk_filter,
  re_model
))

val data = Seq("MRI demonstrated infarction in the upper brain stem , left cerebellum and  right basil ganglia").toDF("text")
val result = trained_pipeline.fit(data).transform(data)

// Show results
result.selectExpr("explode(relations) as relations")
 .select(
   "relations.metadata.chunk1",
   "relations.metadata.entity1",
   "relations.metadata.chunk2",
   "relations.metadata.entity2",
   "relations.result"
 )
 .where("result != 0")
 .show(truncate=false)
+------+---------+-------------+---------------------------+------+
|chunk1|entity1  |chunk2       |entity2                    |result|
+------+---------+-------------+---------------------------+------+
|upper |Direction|brain stem   |Internal_organ_or_component|1     |
|left  |Direction|cerebellum   |Internal_organ_or_component|1     |
|right |Direction|basil ganglia|Internal_organ_or_component|1     |
+------+---------+-------------+---------------------------+------+

```
```python
# Relation Extraction between body parts
# This is a continuation of the RENerChunksFilter example. See that class on how to extract the relation chunks.
# Define the extraction model
re_ner_chunk_filter = RENerChunksFilter() \
 .setInputCols(["ner_chunks", "dependencies"]) \
 .setOutputCol("re_ner_chunks") \
 .setMaxSyntacticDistance(4) \
 .setRelationPairs(["internal_organ_or_component-direction"])

re_model = RelationExtractionDLModel.pretrained("redl_bodypart_direction_biobert", "en", "clinical/models") \
  .setPredictionThreshold(0.5) \
  .setInputCols(["re_ner_chunks", "sentences"]) \
  .setOutputCol("relations")

trained_pipeline = Pipeline(stages=[
  documenter,
  sentencer,
  tokenizer,
  words_embedder,
  pos_tagger,
  clinical_ner_tagger,
  ner_chunker,
  dependency_parser,
  re_ner_chunk_filter,
  re_model
])

data = spark.createDataFrame([["MRI demonstrated infarction in the upper brain stem , left cerebellum and  right basil ganglia"]]).toDF("text")
result = trained_pipeline.fit(data).transform(data)

# Show results
result.selectExpr("explode(relations) as relations") \
 .select(
   "relations.metadata.chunk1",
   "relations.metadata.entity1",
   "relations.metadata.chunk2",
   "relations.metadata.entity2",
   "relations.result"
 ) \
 .where("result != 0") \
 .show(truncate=False)
+------+---------+-------------+---------------------------+------+
|chunk1|entity1  |chunk2       |entity2                    |result|
+------+---------+-------------+---------------------------+------+
|upper |Direction|brain stem   |Internal_organ_or_component|1     |
|left  |Direction|cerebellum   |Internal_organ_or_component|1     |
|right |Direction|basil ganglia|Internal_organ_or_component|1     |
+------+---------+-------------+---------------------------+------+

```
</div>

</details>

</div>

<div class="h3-box" markdown="1">

### RENerChunksFilter

<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/re/RENerChunksFilter.html">API scaladocs</a>

Filters and outputs combinations of relations between extracted entities, for further processing. This annotator is especially useful to create inputs for the RelationExtractionDLModel.

**Input types:** `CHUNK, DEPENDENCY`

**Output type:** `CHUNK`

<details>
<summary><b>Show Example</b></summary>

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}
```scala
// Define pipeline stages to extract entities
val documenter = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentencer = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentences")

val tokenizer = new Tokenizer()
  .setInputCols("sentences")
  .setOutputCol("tokens")

val words_embedder = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols("sentences", "tokens")
  .setOutputCol("embeddings")

val pos_tagger = PerceptronModel.pretrained("pos_clinical", "en", "clinical/models")
  .setInputCols("sentences", "tokens")
  .setOutputCol("pos_tags")

val dependency_parser = DependencyParserModel.pretrained("dependency_conllu", "en")
  .setInputCols("sentences", "pos_tags", "tokens")
  .setOutputCol("dependencies")

val clinical_ner_tagger = MedicalNerModel.pretrained("jsl_ner_wip_greedy_clinical","en","clinical/models")
  .setInputCols("sentences", "tokens", "embeddings")
  .setOutputCol("ner_tags")

val ner_chunker = new NerConverter()
  .setInputCols("sentences", "tokens", "ner_tags")
  .setOutputCol("ner_chunks")

// Define the relation pairs and the filter
val relationPairs = Array("direction-external_body_part_or_region",
                      "external_body_part_or_region-direction",
                      "direction-internal_organ_or_component",
                      "internal_organ_or_component-direction")

val re_ner_chunk_filter = new RENerChunksFilter()
    .setInputCols("ner_chunks", "dependencies")
    .setOutputCol("re_ner_chunks")
    .setMaxSyntacticDistance(4)
    .setRelationPairs(Array("internal_organ_or_component-direction"))

val trained_pipeline = new Pipeline().setStages(Array(
  documenter,
  sentencer,
  tokenizer,
  words_embedder,
  pos_tagger,
  clinical_ner_tagger,
  ner_chunker,
  dependency_parser,
  re_ner_chunk_filter
))

val data = Seq("MRI demonstrated infarction in the upper brain stem , left cerebellum and  right basil ganglia").toDF("text")
val result = trained_pipeline.fit(data).transform(data)

// Show results
result.selectExpr("explode(re_ner_chunks) as re_chunks")
  .selectExpr("re_chunks.begin", "re_chunks.result", "re_chunks.metadata.entity", "re_chunks.metadata.paired_to")
  .show(6, truncate=false)
+-----+-------------+---------------------------+---------+
|begin|result       |entity                     |paired_to|
+-----+-------------+---------------------------+---------+
|35   |upper        |Direction                  |41       |
|41   |brain stem   |Internal_organ_or_component|35       |
|35   |upper        |Direction                  |59       |
|59   |cerebellum   |Internal_organ_or_component|35       |
|35   |upper        |Direction                  |81       |
|81   |basil ganglia|Internal_organ_or_component|35       |
+-----+-------------+---------------------------+---------+

```
```python
# Define pipeline stages to extract entities
documenter = DocumentAssembler() \
  .setInputCol("text") \
  .setOutputCol("document")

sentencer = SentenceDetector() \
  .setInputCols(["document"]) \
  .setOutputCol("sentences")

tokenizer = Tokenizer() \
  .setInputCols(["sentences"]) \
  .setOutputCol("tokens")

words_embedder = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models") \
  .setInputCols(["sentences", "tokens"]) \
  .setOutputCol("embeddings")

pos_tagger = PerceptronModel.pretrained("pos_clinical", "en", "clinical/models") \
  .setInputCols(["sentences", "tokens"]) \
  .setOutputCol("pos_tags")

dependency_parser = DependencyParserModel.pretrained("dependency_conllu", "en") \
  .setInputCols(["sentences", "pos_tags", "tokens"]) \
  .setOutputCol("dependencies")

clinical_ner_tagger = MedicalNerModel.pretrained("jsl_ner_wip_greedy_clinical","en","clinical/models") \
  .setInputCols(["sentences", "tokens", "embeddings"]) \
  .setOutputCol("ner_tags")

ner_chunker = NerConverter() \
  .setInputCols(["sentences", "tokens", "ner_tags"]) \
  .setOutputCol("ner_chunks")

# Define the relation pairs and the filter
relationPairs = ["direction-external_body_part_or_region",
                      "external_body_part_or_region-direction",
                      "direction-internal_organ_or_component",
                      "internal_organ_or_component-direction"]

re_ner_chunk_filter = RENerChunksFilter() \
    .setInputCols(["ner_chunks", "dependencies"]) \
    .setOutputCol("re_ner_chunks") \
    .setMaxSyntacticDistance(4) \
    .setRelationPairs(["internal_organ_or_component-direction"])

trained_pipeline = Pipeline(stages=[
  documenter,
  sentencer,
  tokenizer,
  words_embedder,
  pos_tagger,
  clinical_ner_tagger,
  ner_chunker,
  dependency_parser,
  re_ner_chunk_filter
])

data = spark.createDataFrame([["MRI demonstrated infarction in the upper brain stem , left cerebellum and  right basil ganglia"]]).toDF("text")
result = trained_pipeline.fit(data).transform(data)

# Show results
result.selectExpr("explode(re_ner_chunks) as re_chunks") \
  .selectExpr("re_chunks.begin", "re_chunks.result", "re_chunks.metadata.entity", "re_chunks.metadata.paired_to") \
  .show(6, truncate=False)
+-----+-------------+---------------------------+---------+
|begin|result       |entity                     |paired_to|
+-----+-------------+---------------------------+---------+
|35   |upper        |Direction                  |41       |
|41   |brain stem   |Internal_organ_or_component|35       |
|35   |upper        |Direction                  |59       |
|59   |cerebellum   |Internal_organ_or_component|35       |
|35   |upper        |Direction                  |81       |
|81   |basil ganglia|Internal_organ_or_component|35       |
+-----+-------------+---------------------------+---------+

```
</div>

</details>

</div>

<div class="h3-box" markdown="1">

### SentenceEntityResolver
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/resolution/SentenceEntityResolverApproach.html">Approach scaladocs</a> |
<a href="https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/resolution/SentenceEntityResolverModel.html">Model scaladocs</a>

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
