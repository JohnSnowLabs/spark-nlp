---
layout: docs
header: true
title: Transformers
permalink: /docs/en/transformers
key: docs-transformers
modify_date: "2019-11-01"
use_language_switcher: "Python-Scala-Java"
---

<div class="h3-box" markdown="1">

## DocumentAssembler: Getting data in

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

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```java
import com.johnsnowlabs.nlp.*;
import com.johnsnowlabs.nlp.annotators.*;
import org.apache.spark.ml.Pipeline;
DocumentAssembler documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
    .setCleanupMode("shrink")
```

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

</div></div><div class="h3-box" markdown="1">

## TokenAssembler: Getting data reshaped

This transformer reconstructs a Document type annotation from tokens,
usually after these have been normalized, lemmatized, normalized, spell
checked, etc, in order to use this document annotation in further
annotators.

**Settable parameters are:**

- setInputCol(inputs:Array(String))
- setOutputCol(output:String)
- setPreservePosition(preservePosition:bool): Whether to preserve the actual position of the tokens or reduce them to one space

**Example:**

Refer to the [TokenAssembler](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.TokenAssembler) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```java
TokenAssembler token_assembler = new TokenAssembler()
    .setInputCols(["sentence", "token"])
    .setOutputCol("assembled")
```


```python
token_assembler = TokenAssembler() \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("assembled")
```

```scala
val token_assembler = new TokenAssembler()
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("assembled")
```

</div></div><div class="h3-box" markdown="1">

## Doc2Chunk

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

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

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

</div></div><div class="h3-box" markdown="1">

## Chunk2Doc

Converts a CHUNK type column back into DOCUMENT. Useful when trying to re-tokenize or do further analysis on a CHUNK result.  

**Settable parameters are:**

- setInputCol()
- setOutputCol()

**Example:**

Refer to the [Chunk2Doc](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.Chunk2Doc) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

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

</div></div><div class="h3-box" markdown="1">

## Finisher

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

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

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

</div></div><div class="h3-box" markdown="1">

## EmbeddingsFinisher

This transformer is designed to deal with embedding annotators: `WordEmbeddings`, `BertEmbeddings`, `SentenceEmbeddingd`, and `ChunkEmbeddings`. By using `EmbeddingsFinisher` you can easily transform your embeddings into array of floats or Vectors which are compatible with Spark ML functions such as LDA, K-mean, Random Forest classifier or any other functions that require `featureCol`.

**Settable parameters are:**

- setInputCols()
- setOutputCols()
- setCleanAnnotations(True) -> Whether to remove and cleanup the rest of the annotators (columns)
- setOutputAsVector(False) -> if enabled, it will output the embeddings as Vectors instead of arrays

**Example:**

Refer to the [EmbeddingsFinisher](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.EmbeddingsFinisher) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```java
todo
```

```python
embeddings_finisher = EmbeddingsFinisher() \
            .setInputCols("sentence_embeddings") \
            .setOutputCols("sentence_embeddings_vectors") \
            .setOutputAsVector(True)
```

```scala
val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("sentence_embeddings", "embeddings")
      .setOutputCols("finished_sentence_embeddings", "finished_embeddings")
      .setOutputAsVector(true)
      .setCleanAnnotations(false)
```
</div></div>

## Import Transformers into Spark NLP

Starting with Spark NLP 3.1.0 the support for HuggingFace exported models in equivalent Spark NLP annotators has been extended. Users can now easily use `saved_model` feature in HuggingFace within a few lines of codes and import any BERT, DistilBERT, RoBERTa, and XLM-RoBERTa models to Spark NLP. We will work on the remaining annotators and extend this support to the rest with each release. 

### Compatibility 

**Spark NLP**: The equivalent annotator in Spark NLP

**TF Hub**: Models from [TF Hub](https://tfhub.dev/)

**HuggingFace**: Models from [HuggingFace](https://huggingface.co/models)

**Model Architecture**: Which architecture is compatible with that annotator

**Flags**:
- Fully supported ✅ 
- Partially supported (requires workarounds) ✔️ 
- Under development ❎ 
- Not supported ❌ 

Spark NLP | TF Hub | HuggingFace | Model Architecture
:------------ | :-------------| :-------------| :-------------| 
BertEmbeddings |  ❎  |  ✅  |  BERT - Small BERT - ELECTRA
BertSentenceEmbeddings |  ❎  | ✅   | BERT - Small BERT - ELECTRA
DistilBertEmbeddings|   |  ✅   | DistilBERT
RoBertaEmbeddings |   | ✅   | RoBERTa - DistilRoBERTa
XlmRoBertaEmbeddings |   | ✅   | XLM-RoBERTa
AlbertEmbeddings | ❎  |  ❎  | 
XlnetEmbeddings |   | ❎  | 
ElmoEmbeddings | ❎  |  ❎  | 
UniversalSentenceEncoder |  ❎ |   | 
T5Transformer |   |  ❌ | 
MarianTransformer|   | ❌  | 

### Notebooks

Spark NLP | HuggingFace Notebooks
:------------ | :-------------|
BertEmbeddings |  [HuggingFace in Spark NLP - BERT](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BERT.ipynb)
BertSentenceEmbeddings | [HuggingFace in Spark NLP - BERT Sentence](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BERT%20Sentence.ipynb)
DistilBertEmbeddings| [HuggingFace in Spark NLP - DistilBERT](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20DistilBERT.ipynb)  
RoBertaEmbeddings | [HuggingFace in Spark NLP - RoBERTa](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20RoBERTa.ipynb)  
XlmRoBertaEmbeddings | [HuggingFace in Spark NLP - XLM-RoBERTa](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20XLM-RoBERTa.ipynb)


### Limitations 

- If you are importing models from HuggingFace as Embeddings they must be for `Fill-Mask` task. Meaning you cannot use a model in BertEmbeddings if they were trained or fine-tuned on token/text classification tasks in HuggingFace. They have a different architecture.
- There is a 2G size limitation with loading a TF SavedModel model in Spark NLP. Your model cannot be larger than 2G size or you will see the following error: `Required array size too large. (We are working on going around this Java limitation, however, for the time being, there are some models which are over 2G and they are not compatible)
