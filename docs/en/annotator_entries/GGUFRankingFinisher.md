# GGUFRankingFinisher

The `GGUFRankingFinisher` is a Spark NLP finisher designed to post-process the output of `AutoGGUFReranker`. It provides advanced ranking capabilities including top-k selection, score-based filtering, and normalization.

## Features

- **Top-K Selection**: Select only the top k documents by relevance score
- **Score Thresholding**: Filter documents by minimum relevance score
- **Min-Max Scaling**: Normalize relevance scores to 0-1 range
- **Sorting**: Automatically sorts documents by relevance score in descending order
- **Ranking**: Adds rank metadata to each document

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `inputCols` | `Array[String]` | Name of input annotation columns containing reranked documents | - |
| `outputCol` | `String` | Name of output annotation column containing ranked documents | `"ranked_documents"` |
| `topK` | `Int` | Maximum number of top documents to return (-1 for no limit) | `-1` |
| `minRelevanceScore` | `Double` | Minimum relevance score threshold | `Double.MinValue` |
| `minMaxScaling` | `Boolean` | Whether to apply min-max scaling to normalize scores | `false` |

## Usage

### Basic Usage

```scala
import com.johnsnowlabs.nlp.finisher.GGUFRankingFinisher

val finisher = new GGUFRankingFinisher()
  .setInputCols("reranked_documents")
  .setOutputCol("ranked_documents")
```

### Top-K Selection

```scala
val finisher = new GGUFRankingFinisher()
  .setInputCols("reranked_documents")
  .setOutputCol("ranked_documents")
  .setTopK(5) // Get top 5 most relevant documents
```

### Score Thresholding

```scala
val finisher = new GGUFRankingFinisher()
  .setInputCols("reranked_documents")
  .setOutputCol("ranked_documents")
  .setMinRelevanceScore(0.3) // Only documents with score >= 0.3
```

### Min-Max Scaling

```scala
val finisher = new GGUFRankingFinisher()
  .setInputCols("reranked_documents")
  .setOutputCol("ranked_documents")
  .setMinMaxScaling(true) // Normalize scores to 0-1 range
```

### Combined Usage

```scala
val finisher = new GGUFRankingFinisher()
  .setInputCols("reranked_documents")
  .setOutputCol("ranked_documents")
  .setTopK(3)
  .setMinRelevanceScore(0.2)
  .setMinMaxScaling(true)
```

## Complete Pipeline Example

```scala
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.seq2seq.AutoGGUFReranker
import com.johnsnowlabs.nlp.finisher.GGUFRankingFinisher
import org.apache.spark.ml.Pipeline

// Document assembler
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

// Reranker
val reranker = AutoGGUFReranker
  .pretrained()
  .setInputCols("document")
  .setOutputCol("reranked_documents")
  .setQuery("A man is eating pasta.")

// Finisher
val finisher = new GGUFRankingFinisher()
  .setInputCols("reranked_documents")
  .setOutputCol("ranked_documents")
  .setTopK(3)
  .setMinMaxScaling(true)

// Pipeline
val pipeline = new Pipeline()
  .setStages(Array(documentAssembler, reranker, finisher))
```

## Python Usage

```python
from sparknlp.finisher import GGUFRankingFinisher
from sparknlp.annotator import AutoGGUFReranker
from sparknlp.base import DocumentAssembler
from pyspark.ml import Pipeline

# Create finisher
finisher = GGUFRankingFinisher() \
    .setInputCols("reranked_documents") \
    .setOutputCol("ranked_documents") \
    .setTopK(3) \
    .setMinMaxScaling(True)

# Create pipeline
pipeline = Pipeline(stages=[document_assembler, reranker, finisher])
```

## Output Schema

The finisher produces a DataFrame with the output annotation column containing ranked documents. Each document annotation contains:

- **result**: The document text
- **metadata**: Including `relevance_score`, `rank`, and original `query` information
- **begin/end**: Character positions in the original text
- **annotatorType**: Set to `DOCUMENT`

## Processing Order

The finisher applies operations in the following order:

1. **Extract** documents and metadata from annotations across all rows
2. **Scale** relevance scores (if min-max scaling is enabled)
3. **Filter** by minimum relevance score threshold
4. **Sort** by relevance score (descending)
5. **Limit** to top-k results globally (if specified)
6. **Add rank** metadata to each document
7. **Return** filtered rows with ranked annotations

## Notes

- The finisher expects input from `AutoGGUFReranker` or compatible annotators that produce documents with `relevance_score` metadata
- Min-max scaling is applied before threshold filtering, so thresholds should be set according to the scaled range (0.0-1.0)
- Results are always sorted by relevance score in descending order
- Top-k filtering is applied globally across all input rows, not per row
- The finisher adds `rank` metadata to each document indicating its position in the ranking
- Rows with empty annotation arrays are filtered out from the result
