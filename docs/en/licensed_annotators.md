---
layout: article
title: Licensed Annotators
permalink: /docs/en/licensed_annotators
key: docs-licensed-annotators
modify_date: "2020-08-10"
use_language_switcher: "Python-Scala"
---

## Spark-NLP Licensed

The following annotators are available by buying a John Snow Labs Spark NLP license.
They are mostly meant for healthcare applications but other applications have been made with these NLP features.
Check out www.johnsnowlabs.com for more information.

### AssertionLogReg 
<a href="https://nlp.johnsnowlabs.com/licensed/api/index.html#com.johnsnowlabs.nlp.annotators.assertion.logreg.AssertionLogRegApproach">Estimator scaladocs</a> | 
<a href="https://nlp.johnsnowlabs.com/licensed/api/index.html#com.johnsnowlabs.nlp.annotators.assertion.logreg.AssertionLogRegModel">Transformer scaladocs</a>

It will classify each clinically relevant named entity into its assertion:

type: "present", "absent", "hypothetical", "conditional",
"associated_with_other_person", etc.

**Input types:** `"sentence", "ner_chunk", "embeddings"`

**Output type:** `"assertion"`

**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```python
logRegAssert = AssertionLogRegApproach()
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("pos") \
    .setLabelCol("label") \
    .setMaxIter(26) \
    .setReg(0.00192) \
    .setEnet(0.9) \
    .setBefore(10) \
    .setAfter(10) \
    .setStartCol("start") \
    .setEndCol("end")
```

```scala
val logRegAssert = new AssertionLogRegApproach()
    .setInputCols(Array("sentence", "ner_chunk", "embeddings"))
    .setOutputCol("pos")
    .setLabelCol("label")
    .setMaxIter(26)
    .setReg(0.00192)
    .setEnet(0.9)
    .setBefore(10)
    .setAfter(10)
    .setStartCol("start")
    .setEndCol("end")
```

### AssertionDL 
<a href="https://nlp.johnsnowlabs.com/licensed/api/index.html#com.johnsnowlabs.nlp.annotators.assertion.dl.AssertionDLApproach">Estimator scaladocs</a> | 
<a href="https://nlp.johnsnowlabs.com/licensed/api/index.html#com.johnsnowlabs.nlp.annotators.assertion.dl.AssertionDLModel">Transformer scaladocs</a>

It will classify each clinically relevant named entity into its assertion
type: "present", "absent", "hypothetical", "conditional", "associated_with_other_person", etc.

**Input types:** "sentence", "ner_chunk", "embeddings"

**Output type:** "assertion"

**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```python
dlAssert = AssertionDLApproach() \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("pos") \
    .setGraphFolder("path/to/graphs") \
    .setConfigProtoBytes(b) \
    .setLabelCol("label") \
    .setBatchSize(64) \
    .setEpochs(5) \
    .setLearningRate(0.001) \
    .setDropout(0.05) \
    .setMaxSentLen(250) \
    .setStartCol("start") \
    .setEndCol("end")
```

```scala
val dlAssert = new AssertionDLApproach()
    .setInputCols(Array("sentence", "ner_chunk", "embeddings"))
    .setOutputCol("pos")
    .setGraphFolder("path/to/graphs")
    .setConfigProtoBytes(b)
    .setLabelCol("label")
    .setBatchSize(64)
    .setEpochs(5)
    .setLearningRate(0.001)
    .setDropout(0.05)
    .setMaxSentLen(250)
    .setStartCol("start")
    .setEndCol("end")
```

### Chunk2Token
<a href="https://nlp.johnsnowlabs.com/licensed/api/index.html#com.johnsnowlabs.nlp.annotators.Chunk2Token">Transformer scaladocs</a>

Transforms a complete chunk Annotation into a token Annotation without further tokenization, as opposed to ChunkTokenizer.

**Input types:** "chunk",

**Output type:** "token"

**Example:**

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

### ChunkEntityResolver
<a href="https://nlp.johnsnowlabs.com/licensed/api/index.html#com.johnsnowlabs.nlp.annotators.resolution.ChunkEntityResolverApproach">Estimator scaladocs</a> | 
<a href="https://nlp.johnsnowlabs.com/licensed/api/index.html#com.johnsnowlabs.nlp.annotators.resolution.ChunkEntityResolverModel">Transformer scaladocs</a>

Assigns a standard code (ICD10 CM, PCS, ICDO; CPT) to chunk tokens identified from TextMatchers or the NER Clinical Models and embeddings pooled by ChunkEmbeddings

**Input types:** "chunk_token", "embeddings"

**Output type:** "resolution"

**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```python
resolver = ChunkEntityResolverApproach() \
    .setInputCols(["chunk_token", "chunk_embeddings"]) \
    .setOutputCol("token") \
    .setLabelCol("label") \
    .setNormalizedCol("normalized") \
    .setNeighbours(200) \
    .setAlternatives(25) \
    .setThreshold(4) \
    .setExtramassPenalty(1) \
    .setEnableWmd(True) \
    .setEnableTfidf(True) \
    .setEnableJaccard(True) \
    .setEnableSorensenDice(False) \
    .setEnableJaroWinkler(False) \
    .setEnableLevenshtein(False) \
    .setDistanceWeights([1,3,3,0,0,0]) \
    .setPoolingStrategy("AVERAGE") \
    .setMissAsEmpty(True)
```
```scala
val resolver = new ChunkEntityResolverApproach()
    .setInputCols(Array("chunk_token", "chunk_embeddings"))
    .setOutputCol("token")
    .setLabelCol("label")
    .setNormalizedCol("normalized")
    .setNeighbours(200)
    .setAlternatives(25)
    .setThreshold(4)
    .setExtramassPenalty(1)
    .setEnableWmd(true)
    .setEnableTfidf(true)
    .setEnableJaccard(true)
    .setEnableSorensenDice(false)
    .setEnableJaroWinkler(false)
    .setEnableLevenshtein(false)
    .setDistanceWeights(Array(1,3,3,0,0,0))
    .setPoolingStrategy("AVERAGE")
    .setMissAsEmpty(true)
```

### DocumentLogRegClassifier

A convenient TFIDF-LogReg classifier that accepts "token" input type and outputs "selector"; an input type mainly used in RecursivePipelineModels

**Input types:** "token"

**Output type:** "category"

**Example:**

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

### DeIdentificator

Identifies potential pieces of content with personal information about
patients and remove them by replacing with semantic tags.

**Input types:** "sentence", "token", "ner_chunk"

**Output type:** "deidentified"

**Functions:**

- setRegexPatternsDictionary(path, read_as, options)

### Contextual Parser

This annotator provides Regex + Contextual Matching, based on a JSON file.
**Output type:** "sentence", "token"  
**Input types:** "chunk"  
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

### References

[1] Speech and Language Processing. Daniel Jurafsky & James H. Martin. 2018


### RelationExtraction 
<a href="https://nlp.johnsnowlabs.com/licensed/api/index.html#com.johnsnowlabs.nlp.annotators.re.RelationExtractionApproach">Estimator scaladocs</a> | 
<a href="https://nlp.johnsnowlabs.com/licensed/api/index.html#com.johnsnowlabs.nlp.annotators.re.RelationExtractionModel">Transformer scaladocs</a>

Extracts and classifier instances of relations between named entities.

**Input types:** "pos", "ner_chunk", "embeddings", "dependency"

**Output type:** "category"

**Example:**

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
