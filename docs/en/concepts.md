---
layout: article
title: Concepts
permalink: /docs/en/concepts
key: docs-concepts
modify_date: "2019-10-23"
use_language_switchter: true

---

## Annotators Guideline

### Concepts

### Spark NLP Imports

We attempt making necessary imports easy to reach, **base** will include
general Spark NLP transformers and concepts, while **annotator** will
include all annotators that we currently provide. **embeddings** include
word embedding annotators. This does not include Spark imports.

**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```python
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.embeddings import *
```

```scala
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
```

### Spark ML Pipelines

SparkML Pipelines are a uniform structure that helps creating and tuning
practical machine learning pipelines. Spark NLP integrates with them
seamlessly so it is important to have this concept handy. Once a
**Pipeline** is trained with **fit()**, this becomes a **PipelineModel**  

**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```python
from pyspark.ml import Pipeline
pipeline = Pipeline().setStages([...])
```

```scala
import org.apache.spark.ml.Pipeline
new Pipeline().setStages(Array(...))
```

### LightPipeline

LightPipelines are Spark ML pipelines converted into a single machine
but multithreaded task, becoming more than 10x times faster for smaller
amounts of data (small is relative, but 50k sentences is roughly a good
maximum). To use them, simply plug in a trained (fitted) pipeline.

**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```python
from sparknlp.base import LightPipeline
LightPipeline(someTrainedPipeline).annotate(someStringOrArray)
```

```scala
import com.johnsnowlabs.nlp.LightPipeline
new LightPipeline(somePipelineModel).annotate(someStringOrArray))
```

**Functions:**

- annotate(string or string\[\]): returns dictionary list of annotation
results
- fullAnnotate(string or string\[\]): returns dictionary list of entire
annotations content

### RecursivePipeline

Recursive pipelines are SparkNLP specific pipelines that allow a Spark
ML Pipeline to know about itself on every Pipeline Stage task, allowing
annotators to utilize this same pipeline against external resources to
process them in the same way the user decides. Only some of our
annotators take advantage of this. RecursivePipeline behaves exactly
the same than normal Spark ML pipelines, so they can be used with the
same intention.

**Example:**

{% include programmingLanguageSelectScalaPython.html %}

```python
from sparknlp.annotator import *
recursivePipeline = RecursivePipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        lemmatizer,
        finisher
        ])
```

```scala
import com.johnsnowlabs.nlp.RecursivePipeline
val recursivePipeline = new RecursivePipeline()
        .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        lemmatizer,
        finisher
        ))
```

### EmbeddingsHelper

#### Deal with word embeddings

Allows loading, saving and setting word embeddings for annotators.

An embeddings reference, or embeddingsRef, is a user-given name for
annotators to lookup the embeddings database. Since Spark NLP 2.0,
embeddings are annotators on its own, however, certain use cases may
require multiple embedding annotators, and you might not want to
duplicate the database on all of them. Hence, you can use reference in
combination with the param `setIncludeEmbeddings(false)` to refer to the
same database without loading them.

In the future, some annotators might also need random access to the
embeddings database, so they might take an embeddingsRef, apart from the
pipeline annotator.

This applies only to `WordEmbeddings` not `BertEmbeddings`.

**Functions:**

- load(path, spark, format, reference, dims, caseSensitive) -> Loads
embeddings from disk in any format possible: 'TEXT', 'BINARY',
'SPARKNLP'. Makes embeddings available for Annotators without included
embeddings.
- save(path, embeddings, spark) -> Saves provided embeddings to path,
using current SparkSession

#### Annotator with Word Embeddings

Some annotators use word embeddings. This is a common functionality
within them. Since Spark NLP 2.0, embeddings as annotator means the
rest annotators don't use this interface anymore, however, for
developers reference, they still exist and might be used in annotators
that require random access to word embeddings.

These functions are included in the embedding annotators 
`WordEmbeddings` and `BertEmbeddings`

**Functions (not all of them listed):**

- setIncludeEmbeddings(bool) -> Param to define whether or not to
include word embeddings when saving this annotator to disk (single or
within pipeline)
- setEmbeddingsRef(ref) -> Set whether to use annotators under the
provided name. This means these embeddings will be lookup from the cache
by the ref name. This allows multiple annotators to utilize same word
embeddings by ref name.

### Params and Features

#### Annotator parameters

SparkML uses ML Params to store pipeline parameter maps. In SparkNLP,
we also use Features, which are a way to store parameter maps that are
larger than just a string or a boolean. These features are serialized
as either Parquet or RDD objects, allowing much faster and scalable
annotator information. Features are also broadcasted among executors for
better performance.  
