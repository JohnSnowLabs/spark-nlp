---
layout: article
title: Transformers
permalink: /docs/en/transformers
key: docs-transformers
modify_date: "2019-10-23"
---

## Transformers Guideline

### DocumentAssembler: Getting data in

In order to get through the NLP process, we need to get raw data
annotated. There is a special transformer that does this for us: it
creates the first annotation of type Document which may be used by
annotators down the road. It can read either a String column or an
Array\[String\]  

**Settable parameters are:**

- setInputCol()
- setOutputCol()
- setIdCol() -> OPTIONAL: Sring type column with id information
- setMetadataCol() -> OPTIONAL: Map type column with metadata
information
- setCleanupMode(disabled) -> Cleaning up options, possible values: 
  - disabled: Source kept as original. 
  - inplace: removes new lines and tabs.
  - inplace_full: removes new lines and tabs but also those which were
  converted to strings (i.e. \\n)
  - shrink: removes new lines and tabs, plus merging multiple spaces
  and blank lines to a single space.
  - shrink_full: removews new lines and tabs, including stringified
  values, plus shrinking spaces and blank lines. 

**Example:**

Refer to the [DocumentAssembler](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.DocumentAssembler)
Scala docs for more details on the API.

```python
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from pyspark.ml import Pipeline
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document") \
    .setCleanupMode("shrink")
```

```scala
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators._
import org.apache.spark.ml.Pipeline
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
    .setCleanupMode("shrink")
```

### TokenAssembler: Getting data reshaped

This transformer reconstructs a Document type annotation from tokens,
usually after these have been normalized, lemmatized, normalized, spell
checked, etc, in order to use this document annotation in further
annotators.

**Settable parameters are:**

- setInputCol()
- setOutputCol()

**Example:**

Refer to the [TokenAssembler](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.TokenAssembler) Scala docs for more details on the API.

```python
token_assembler = TokenAssembler() \
    .setInputCols(["normalized"]) \
    .setOutputCol("assembled")
```

```scala
val token_assembler = new TokenAssembler()
    .setInputCols("normalized")
    .setOutputCol("assembled")
```

### Doc2Chunk

Converts DOCUMENT type annotations into CHUNK type with the contents of a chunkCol. Chunk text must be contained within input DOCUMENT. May be either StringType or ArrayType\[StringType\] (using isArray Param) Useful for annotators that require a CHUNK type input.  

**Settable parameters are:**

- setInputCol()
- setOutputCol()
- setIsArray(bool) -> Whether the target chunkCol is `ArrayType<StringType>`
- setChunkCol(string) -> String or StringArray column with the chunks that belong to the `inputCol` target
- setStartCol(string) -> Target INT column pointing to the token index (split by white space)
- setStartColByTokenIndex(bool) -> Whether to use token index by whitespace or character index in `startCol`
- setFailOnMissing(bool) -> Whether to fail when a chunk is not found within inputCol
- setLowerCase(bool) -> whether to increase matching by lowercasing everything before matching

**Example:**

Refer to the [Doc2Chunk](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.Doc2Chunk) Scala docs for more details on the API.

```python
chunker = Doc2Chunk()\
    .setInputCols(["document"])\
    .setOutputCol("chunk")\
    .setIsArray(False)\
    .setChunkCol("some_column")
```

```scala
val chunker = new Doc2Chunk()
    .setInputCols("document")
    .setOutputCol("chunk")
    .setIsArray(false)
    .setChunkCol("some_column")
```

### Chunk2Doc

Converts a CHUNK type column back into DOCUMENT. Useful when trying to re-tokenize or do further analysis on a CHUNK result.  

**Settable parameters are:**

- setInputCol()
- setOutputCol()

**Example:**

Refer to the [Chunk2Doc](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.Chunk2Doc) Scala docs for more details on the API.

```python
chunk_doc = Chunk2Doc()\
    .setInputCols(["chunk_output"])\
    .setOutputCol("new_document")\
```

```scala
val chunk_doc = new Chunk2Doc()
    .setInputCols("chunk_output")
    .setOutputCol("new_document")
```

### Finisher

Once we have our NLP pipeline ready to go, we might want to use our annotation results somewhere else where it is easy to use. The Finisher outputs annotation(s) values into string.

**Settable parameters are:**

- setInputCols()
- setOutputCols()
- setCleanAnnotations(True) -> Whether to remove intermediate annotations
- setValueSplitSymbol("#") -> split values within an annotation character
- setAnnotationSplitSymbol("@") -> split values between annotations character
- setIncludeMetadata(False) -> Whether to include metadata keys. Sometimes useful in some annotations
- setOutputAsArray(False) -> Whether to output as Array. Useful as input for other Spark transformers.

**Example:**

Refer to the [Finisher](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.Finisher) Scala docs for more details on the API.

```python
finisher = Finisher() \
    .setInputCols(["token"]) \
    .setIncludeMetadata(True) # set to False to remove metadata
```

```scala
val finisher = new Finisher()
    .setInputCols("token")
    .setIncludeMetadata(true) // set to False to remove metadata
```
