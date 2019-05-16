---
layout: article
title: Annotators
permalink: /docs/en/annotators
key: docs-annotators
modify_date: "2019-05-16"
---

## Concepts

### Spark NLP Imports

Since version 1.5.0 we are making necessary imports easy to reach, **base** will include general Spark NLP transformers and concepts, while **annotator** will include all annotators that we currently provide. We also need SparkML pipelines.

**Example:**

{% highlight python %}
from sparknlp.base import *
from sparknlp.annotator import *
{% endhighlight %}

{% highlight scala %}
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
{% endhighlight %}

### Spark ML Pipelines

SparkML Pipelines are a uniform structure that helps creating and tuning practical machine learning pipelines. Spark NLP integrates with them seamlessly so it is important to have this concept handy. Once a **Pipeline** is trained with **fit()**, this becomes a **PipelineModel**  

**Example:**

{% highlight python %}
from pyspark.ml import Pipeline
pipeline = Pipeline().setStages([...])
{% endhighlight %}

{% highlight scala %}
import org.apache.spark.ml.Pipeline
new Pipeline().setStages(Array(...))
{% endhighlight %}

### LightPipeline

#### A super-fast Spark-NLP pipeline for small data

LightPipelines are Spark ML pipelines converted into a single machine but multithreaded task, becoming more than 10x times faster for smaller amounts of data (50k lines of text or below). To use them, simply plug in a trained (fitted) pipeline.  
**Example:**

{% highlight python %}
from sparknlp.base import LightPipeline
LightPipeline(someTrainedPipeline).annotate(someStringOrArray)
{% endhighlight %}

{% highlight scala %}
import com.johnsnowlabs.nlp.LightPipeline
new LightPipeline(somePipelineModel).annotate(someStringOrArray))
{% endhighlight %}

**Functions:**

- annotate(string or string\[\]): returns dictionary list of annotation results
- fullAnnotate(string or string\[\]): returns dictionary list of entire annotations content

### RecursivePipeline

#### A smarter Spark-NLP pipeline

Recursive pipelines are SparkNLP specific pipelines that allow a Spark ML Pipeline to know about itself on every Pipeline Stage task, allowing annotators to utilize this same pipeline against external resources to process them in the same way the user decides. Only some of our annotators take advantage of this. RecursivePipeline behaves exactly the same than normal Spark ML pipelines, so they can be used with the same intention.  
**Example:**

{% highlight python %}
from sparknlp.annotator import *
recursivePipeline = RecursivePipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        lemmatizer,
        finisher
        ])
{% endhighlight %}

{% highlight scala %}
import com.johnsnowlabs.nlp.RecursivePipeline
val recursivePipeline = new RecursivePipeline()
        .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        lemmatizer,
        finisher
        ))
{% endhighlight %}

### ExternalResource

#### Data properties outside the pipeline

ExternalResource represent the properties of external data to be read, usually by the ResourceHelper (which is explained below). It contains information into how such external source may be read, and allows different protocols (hdfs, s3, etc) and formats (csv, text, parquet, etc). User does not usually need to create explicitly External Resources, but function parameters usually ask for elements used by it.  

**Arguments:**

- path -> Takes a path with protocol of desintation file or folder
- ReadAs -> "LINE_BY_LINE" or "SPARK_DATASET" will tell SparkNLP to use spark or not for this file or folder
- options -> Contains information passed to Spark reader (e.g. format: "text") and other useful options for annotators (e.g. delimiter)

**Example:**

{% highlight python %}
regex_matcher = RegexMatcher() \
    .setStrategy("MATCH_ALL") \
    .setExternalRules(path="/some/path", delimiter=",", read_as=ReadAs.LINE_BY_LINE, options={"format": "parquet"}) \
    .setOutputCol("regex")
{% endhighlight %}

### ResourceHelper

#### Deal with data outside the pipeline

When working with external resources, like training data that is not part of the pipeline process, our annotators use the ResourceHelper to efficiently parse and extract data into specific formats. This class may be utilized for other purposes by the user (Only in Scala)

When working with external resources, like training data that is not part of the pipeline process, our annotators use the ResourceHelper to efficiently parse and extract data into specific formats. This class may be utilized for other purposes by the user.

**Functions (not all of them listed):**

- createDatasetFromText(path, includeFilename, includeRowNumber, aggregateByFile) -> Takes file or folder and builds up an aggregated dataset
- parseKeyValueText(externalResource) -> Parses delimited text with delimiter
- parseLines(externalResource) -> Parses line by line text
- parseTupleText(externalResource) -> Parses a text as a delimited tuple
- parseTupleSentences(externalResource) -> Parses tagged tokens with a specific delimiter
- wordCount(externalResources) -> Counts appearance of each word in text

### EmbeddingsHelper

#### Deal with word embeddings

Allows loading, saving and setting word embeddings for annotators

**Functions (not all of them listed):**

- load(path, spark, format, reference, dims, caseSensitive) -> Loads embeddings from disk in any format possible: 'TEXT', 'BINARY', 'SPARKNLP'. Makes embeddings available for Annotators without included embeddings.
- save(path, embeddings, spark) -> Saves provided embeddings to path, using current SparkSession

#### Annotator with Word Embeddings

Some annotators use word embeddings. This is a common functionality within them.

**Functions (not all of them listed):**

- setIncludeEmbeddings(bool) -> Param to define whether or not to include word embeddings when saving this annotator to disk (single or within pipeline)
- setEmbeddingsRef(ref) -> Set whether to use annotators under the provided name. This means these embeddings will be lookup from the cache by the ref name. This allows multiple annotators to utilize same word embeddings by ref name.

Some annotators use word embeddings. This is a common functionality within them.

### Params and Features

#### Annotator parameters

SparkML uses ML Params to store pipeline parameter maps. In SparkNLP, we also use Features, which are a way to store parameter maps that are larger than just a string or a boolean. These features are serialized as either Parquet or RDD objects, allowing much faster and scalable annotator information. Features are also broadcasted among executors for better performance.  

## Transformers

### DocumentAssembler

#### Getting data in

In order to get through the NLP process, we need to get raw data annotated. There is a special transformer that does this for us: it creates the first annotation of type Document which may be used by annotators down the road. It can read either a String column or an Array\[String\]  

**Settable parameters are:**

- setInputCol()
- setOutputCol()
- setIdCol() -> OPTIONAL: Sring type column with id information
- setMetadataCol() -> OPTIONAL: Map type column with metadata information
- setTrimAndClearNewLines(bool) -> Whether to remove new line characters and trim strings. Defaults to true. Useful for later sentence detection if contains multiple lines.

**Example:**

{% highlight python %}
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from pyspark.ml import Pipeline
documentAssembler = new DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
{% endhighlight %}

{% highlight scala %}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators._
import org.apache.spark.ml.Pipeline
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
{% endhighlight %}

### TokenAssembler

#### Getting data reshaped

This transformer reconstructs a Document type annotation from tokens, usually after these have been normalized, lemmatized, normalized, spell checked, etc, in order to use this document annotation in further annotators.

**Settable parameters are:**

- setInputCol()
- setOutputCol()

**Example:**

{% highlight python %}
token_assembler = TokenAssembler() \
    .setInputCols(["normalized"]) \
    .setOutputCol("assembled")
{% endhighlight %}

{% highlight scala %}
val token_assembler = new TokenAssembler()
    .setInputCols("normalized")
    .setOutputCol("assembled")
{% endhighlight %}

### Doc2Chunk

Converts DOCUMENT type annotations into CHUNK type with the contents of a chunkCol. Chunk text must be contained within input DOCUMENT. May be either StringType or ArrayType\[StringType\] (using isArray Param) Useful for annotators that require a CHUNK type input.  

**Settable parameters are:**

- setInputCol()
- setOutputCol()
- setIsArray()
- setChunkCol()

**Example:**

{% highlight python %}
chunker = Doc2Chunk()\
    .setInputCols(["document"])\
    .setOutputCol("chunk")\
    .setIsArray(False)\
    .setChunkCol("some_column")
{% endhighlight %}

{% highlight scala %}
val chunker = new Doc2Chunk()
    .setInputCols("document")
    .setOutputCol("chunk")
    .setIsArray(false)
    .setChunkCol("some_column")
{% endhighlight %}

### Chunk2Doc

Converts a CHUNK type column back into DOCUMENT. Useful when trying to re-tokenize or do further analysis on a CHUNK result.  

**Settable parameters are:**

- setInputCol()
- setOutputCol()

**Example:**

{% highlight python %}
chunk_doc = Chunk2Doc()\
    .setInputCols(["chunk_output"])\
    .setOutputCol("new_document")\
{% endhighlight %}

{% highlight scala %}
val chunk_doc = new Chunk2Doc()
    .setInputCols("chunk_output")
    .setOutputCol("new_document")
{% endhighlight %}

### Finisher

#### Getting data out

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

{% highlight python %}
finisher = Finisher() \
    .setInputCols(["sentiment"]) \
    .setIncludeMetadata(True)
{% endhighlight %}

{% highlight scala %}
val finisher = new Finisher()
    .setInputCols("token")
    .setIncludeMetadata(true)
{% endhighlight %}

## Training Datasets

### POS Dataset

In order to train a Part of Speech Tagger annotator, we need to get corpus data as a spark dataframe. There is a component that does this for us: it reads a plain text file and transforms it to a spark dataset.  

**Input File Format:**

```bash
A|DT few|JJ months|NNS ago|RB you|PRP received|VBD a|DT letter|NN
```

**Available parameters are:**

- spark: Spark session
- path(string): Path to file with corpus data for training POS
- delimiter(string): Delimiter of token and postag. Defaults to `|`
- outputPosCol(string): Name of the column with POS values. Defaults to "tags".

**Example:**  

{% highlight python %}
from sparknlp.training import POS
train_pos = POS().readDataset(spark, "./src/main/resources/anc-pos-corpus")
{% endhighlight %}

{% highlight scala %}
import com.johnsnowlabs.nlp.training.POS
val trainPOS = POS().readDataset(spark, "./src/main/resources/anc-pos-corpus")
{% endhighlight %}

### CoNLL Dataset

In order to train a Named Entity Recognition DL annotator, we need to get CoNLL format data as a spark dataframe. There is a component that does this for us: it reads a plain text file and transforms it to a spark dataset.

**Available parameters are:**

- spark: Spark session
- path(string): Path to a [CoNLL 2003 IOB NER file](https://www.clips.uantwerpen.be/conll2003/ner).
- readAs(string): Can be LINE_BY_LINE or SPARK_DATASET, with options if latter is used (default LINE_BY_LINE)

**Example:**

{% highlight python %}
from sparknlp.training import CoNLL
training_conll = CoNLL().readDataset(spark, "./src/main/resources/conll2003/eng.train")
{% endhighlight %}

{% highlight scala %}
import com.johnsnowlabs.nlp.training.CoNLL
val trainingConll = CoNLL().readDataset(spark, "./src/main/resources/conll2003/eng.train")
{% endhighlight %}

### Spell Checkers Dataset

In order to train a Norvig or Symmetric Spell Checkers, we need to get corpus data as a spark dataframe. We can read a plain text file and transforms it to a spark dataset.  

**Example:**

{% highlight python %}
train_corpus = spark.read.text("./sherlockholmes.txt")
                    .withColumnRenamed("value", "text")
{% endhighlight %}

{% highlight scala %}
val trainCorpus = spark.read.text("./sherlockholmes.txt")
                       .select(trainCorpus.col("value").as("text"))
{% endhighlight %}

## Annotators

### Tokenizer

Identifies tokens with tokenization open standards. A few rules will help customizing it if defaults do not fit user needs.  
**Type:** Token  
**Requires:** Document  
**Functions:**

- setTargetPattern: Basic regex rule to identify a candidate for tokenization. Defaults to `\\S+` which means anything not a space
- setSuffixPattern: Regex to identify subtokens that are in the end of the token. Regex has to end with `\\z` and must contain groups (). Each group will become a separate token within the prefix. Defaults to non-letter characters. e.g. quotes or parenthesis
- setPrefixPattern: Regex to identify subtokens that come in the beginning of the token. Regex has to start with `\\A` and must contain groups (). Each group will become a separate token within the prefix. Defaults to non-letter characters. e.g. quotes or parenthesis
- setExtensionPatterns: Array of Regex with groups () to match subtokens within the target pattern. Every group () will become its own separate token. Order matters (later rules will apply first). Its default rules should cover most cases, e.g. part-of-speech as single token
- addInfixPattern: Add an extension pattern regex with groups to the top of the rules (will target first, from more specific to the more general).
- setCompositeTokensPatterns: Adds a list of compound words to mark for ignore. E.g., adding "New York" so it doesn't get split into "New" and "York".

**Note:** all these APIs receive regular expressions so please make sure that you escape special characters according to Java conventions.  

**Example:**

{% highlight python %}
tokenizer = Tokenizer() \
    .setInputCols(["sentences"]) \
    .setOutputCol("token") \
    .addInfixPattern("(\p{L}+)(n't\b)")
{% endhighlight %}

{% highlight scala %}
val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")
    .addInfixPattern("(\p{L}+)(n't\b)")
{% endhighlight %}

### Normalizer

#### Text cleaning

Removes all dirty characters from text following a regex pattern and transforms words based on a provided dictionary  
**Type:** Token  
**Requires:** Token  
**Functions:**

- setPatterns(patterns): Regular expressions list for normalization, defaults \[^A-Za-z\]
- setLowercase(value): lowercase tokens, default true
- setSlangDictionary(path): txt file with delimited words to be transformed into something else

**Example:**

{% highlight python %}
normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")
{% endhighlight %}

{% highlight scala %}
val normalizer = new Normalizer()
    .setInputCols(Array("token"))
    .setOutputCol("normalized")
{% endhighlight %}

### Stemmer

Returns hard-stems out of words with the objective of retrieving the meaningful part of the word  
**Type:** Token  
**Requires:** Token  
**Example:**

{% highlight python %}
stemmer = Stemmer() \
    .setInputCols(["token"]) \
    .setOutputCol("stem")
{% endhighlight %}

{% highlight scala %}
val stemmer = new Stemmer()
    .setInputCols(Array("token"))
    .setOutputCol("stem")
{% endhighlight %}

### Lemmatizer

Retrieves lemmas out of words with the objective of returning a base dictionary word  
**Type:** Token  
**Requires:** Token  
**Input:** abduct -> abducted abducting abduct abducts  
**Functions:** --

- setDictionary(path, keyDelimiter, valueDelimiter, readAs, options): Path and options to lemma dictionary, in lemma vs possible words format. readAs can be LINE_BY_LINE or SPARK_DATASET. options contain option passed to spark reader if readAs is SPARK_DATASET.

**Example:**

{% highlight python %}
lemmatizer = Lemmatizer() \
    .setInputCols(["token"]) \
    .setOutputCol("lemma") \
    .setDictionary("./lemmas001.txt")
{% endhighlight %}

{% highlight scala %}
val lemmatizer = new Lemmatizer()
    .setInputCols(Array("token"))
    .setOutputCol("lemma")
    .setDictionary("./lemmas001.txt")
{% endhighlight %}

### RegexMatcher

Uses a reference file to match a set of regular expressions and put them inside a provided key. File must be comma separated.  
**Type:** Regex  
**Requires:** Document  
**Input:** `the\\s\\w+`, "followed by 'the'"  
**Functions:**

- setStrategy(strategy): Can be any of `MATCH_FIRST|MATCH_ALL|MATCH_COMPLETE`
- setRulesPath(path, delimiter, readAs, options): Path to file containing a set of regex,key pair. readAs can be LINE_BY_LINE or SPARK_DATASET. options contain option passed to spark reader if readAs is SPARK_DATASET.

**Example:**

{% highlight python %}
regex_matcher = RegexMatcher() \
    .setStrategy("MATCH_ALL") \
    .setOutputCol("regex")
{% endhighlight %}

{% highlight scala %}
val regexMatcher = new RegexMatcher()
    .setStrategy(strategy)
    .setInputCols(Array("document"))
    .setOutputCol("regex")
{% endhighlight %}

### TextMatcher

#### Phrase matching

Annotator to match entire phrases provided in a file against a Document  
**Type:** Entity  
**Requires:** Document  
**Input:** hello world, I am looking for you  
**Functions:**

- setEntities(path, format, options): Provides a file with phrases to match. Default: Looks up path in configuration.  
- path: a path to a file that contains the entities in the specified format.  
- readAs: the format of the file, can be one of {ReadAs.LINE_BY_LINE, ReadAs.SPARK_DATASET}. Defaults to LINE_BY_LINE.  
- options: a map of additional parameters. Defaults to {"format": "text"}.

**Example:**

{% highlight python %}
entity_extractor = TextMatcher() \
    .setInputCols(["inputCol"])\
    .setOutputCol("entity")\
    .setEntities("/path/to/file/myentities.txt")
{% endhighlight %}

{% highlight scala %}
val entityExtractor = new TextMatcher()
    .setInputCols("inputCol")
    .setOutputCol("entity")
    .setEntities("/path/to/file/myentities.txt")
{% endhighlight %}

### Chunker

#### Meaningful phrase matching

This annotator matches a pattern of part-of-speech tags in order to return meaningful phrases from document

**Type:** Document  
**Requires:** Document  
**Functions:**

- setRegexParsers(patterns): A list of regex patterns to match chunks, for example: Array("‹DT›?‹JJ›\*‹NN›")
- addRegexParser(patterns): adds a pattern to the current list of chunk patterns, for example: "‹DT›?‹JJ›\*‹NN›"

**Example:**

{% highlight python %}
chunker = Chunker() \
    .setInputCols(["pos"]) \
    .setOutputCol("chunk") \
    .setRegexParsers(["‹NNP›+", "‹DT|PP\\$›?‹JJ›*‹NN›"])
{% endhighlight %}

{% highlight scala %}
val chunker = new Chunker()
    .setInputCols(Array("pos"))
    .setOutputCol("chunks")
    .setRegexParsers(Array("‹NNP›+", "‹DT|PP\\$›?‹JJ›*‹NN›"))
{% endhighlight %}

### DateMatcher

#### Date-time parsing

Reads from different forms of date and time expressions and converts them to a provided date format. Extracts only ONE date per sentence. Use with sentence detector for more matches.  
**Type:** Date  
**Requires:** Document  
**Reads the following kind of dates:**

- 1978-01-28
- 1984/04/02
- 1/02/1980
- 2/28/79
- The 31st of April in the year 2008
- Fri, 21 Nov 1997
- Jan 21, '97
- Sun, Nov 21
- jan 1st
- next thursday
- last wednesday
- today
- tomorrow
- yesterday
- next week
- next month
- next year
- day after
- the day before
- 0600h
- 06:00 hours
- 6pm
- 5:30 a.m.
- at 5
- 12:59
- 23:59
- 1988/11/23 6pm
- next week at 7.30
- 5 am tomorrow

**Functions:**

- setDateFormat(format): SimpleDateFormat standard date formatting. Defaults to yyyy/MM/dd

**Example:**

{% highlight python %}
date_matcher = DateMatcher() \
    .setOutputCol("date") \
    .setDateFormat("yyyyMM")
{% endhighlight %}

{% highlight scala %}
val dateMatcher = new DateMatcher()
    .setFormat("yyyyMM")
    .setOutputCol("date")
{% endhighlight %}

### SentenceDetector

#### Sentence Boundary Detector

Finds sentence bounds in raw text. Applies rules from Pragmatic Segmenter.  
**Type:** Document  
**Requires:** Document  
**Functions:**

- setCustomBounds(string): Custom sentence separator text
- setUseCustomOnly(bool): Use only custom bounds without considering those of Pragmatic Segmenter. Defaults to false. Needs customBounds.
- setUseAbbreviations(bool): Whether to consider abbreviation strategies for better accuracy but slower performance. Defaults to true.
- setExplodeSentences(bool): Whether to split sentences into different Dataset rows. Useful for higher parallelism in fat rows. Defaults to false.

**Example:**

{% highlight python %}
sentence_detector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")
{% endhighlight %}

{% highlight scala %}
val sentenceDetector = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentence")
{% endhighlight %}

### DeepSentenceDetector

#### Sentence Boundary Detector with Machine Learning

Finds sentence bounds in raw text. Applies a Named Entity Recognition DL model.  
**Type:** Document  
**Requires:** Document, Token, Chunk  
**Functions:**

- setIncludePragmaticSegmenter(bool): Whether to include rule-based sentence detector as first filter. Defaults to false.
- setEndPunctuation(patterns): An array of symbols that deep sentence detector will consider as an end of sentence punctuation. Defaults to ".", "!", "?"

**Example:**

{% highlight python %}
deep_sentence_detector = DeepSentenceDetector() \
    .setInputCols(["document", "token", "ner_con"]) \
    .setOutputCol("sentence") \
    .setIncludePragmaticSegmenter(True) \
    .setEndPunctuation([".", "?"])
{% endhighlight %}

{% highlight scala %}
val deepSentenceDetector = new DeepSentenceDetector()
    .setInputCols(Array("document", "token", "ner_con"))
    .setOutputCol("sentence")
    .setIncludePragmaticSegmenter(true)
    .setEndPunctuation(Array(".", "?"))
{% endhighlight %}

### POSTagger

#### Part of speech tagger

Sets a POS tag to each word within a sentence. Its train data (train_pos) is a spark dataset of [POS format values](#TrainPOS) with Annotation columns.  
**Type:** POS  
**Requires:** Document, Token  
**Functions:**

- setNIterations(number): Number of iterations for training. May improve accuracy but takes longer. Default 5.
- setPosColumn(colname): Column containing an array of POS Tags matching every token on the line.

**Example:**

{% highlight python %}
pos_tagger = PerceptronApproach() \
    .setInputCols(["token", "sentence"]) \
    .setOutputCol("pos") \
    .setIterations(2) \
    .fit(train_pos)
{% endhighlight %}

{% highlight scala %}
val posTagger = new PerceptronApproach()
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("pos")
    .setIterations(2)
    .fit(trainPOS)
{% endhighlight %}

### ViveknSentimentDetector

#### Sentiment analysis

Scores a sentence for a sentiment  
**Type:** sentiment  
**Requires:** Document, Token  
**Functions:**

- setSentimentCol(colname): Column with sentiment analysis row's result for training. If not set, external sources need to be set instead.
- setPositiveSource(path, tokenPattern, readAs, options): Path to file or folder with positive sentiment text, with tokenPattern the regex pattern to match tokens in source. readAs either LINE_BY_LINE or as SPARK_DATASET. If latter is set, options is passed to reader
- setNegativeSource(path, tokenPattern, readAs, options): Path to file or folder with positive sentiment text, with tokenPattern the regex pattern to match tokens in source. readAs either LINE_BY_LINE or as SPARK_DATASET. If latter is set, options is passed to reader
- setPruneCorpus(true): when training on small data you may want to disable this to not cut off infrequent words

**Input:** File or folder of text files of positive and negative data  
**Example:**

{% highlight python %}
sentiment_detector = SentimentDetector() \
    .setInputCols(["lemma", "sentence"]) \
    .setOutputCol("sentiment")
{% endhighlight %}

{% highlight scala %}
val sentimentDetector = new ViveknSentimentApproach()
        .setInputCols(Array("token", "sentence"))
        .setOutputCol("vivekn")
        .setPositiveSourcePath("./positive/1.txt")
        .setNegativeSourcePath("./negative/1.txt")
        .setCorpusPrune(false)
{% endhighlight %}

### SentimentDetector: Sentiment analysis

Scores a sentence for a sentiment  
**Type:** sentiment  
**Requires:** Document, Token  
**Functions:**

- setDictionary(path, delimiter, readAs, options): path to file with list of inputs and their content, with such delimiter, readAs LINE_BY_LINE or as SPARK_DATASET. If latter is set, options is passed to spark reader.
- setPositiveMultiplier(double): Defaults to 1.0
- setNegativeMultiplier(double): Defaults to -1.0
- setIncrementMultiplier(double): Defaults to 2.0
- setDecrementMultiplier(double): Defaults to -2.0
- setReverseMultiplier(double): Defaults to -1.0

**Input:**

- superb,positive
- bad,negative
- lack of,revert
- very,increment
- barely,decrement

**Example:**

{% highlight python %}
sentiment_detector = SentimentDetector() \
    .setInputCols(["lemma", "sentence"]) \
    .setOutputCol("sentiment")
{% endhighlight %}

{% highlight scala %}
val sentimentDetector = new SentimentDetector
    .setInputCols(Array("token", "sentence"))
    .setOutputCol("sentiment")
{% endhighlight %}

### Word Embeddings

Word Embeddings lookup annotator that maps tokens to vectors  
**Type:** Word_Embeddings  
**Requires:** Document, Token  
**Functions:**

- setEmbeddingsSource:(path, nDims, format) - sets [word embeddings](https://en.wikipedia.org/wiki/Word_embedding) options. path - word embeddings file nDims - number of word embeddings dimensions format - format of word embeddings files:  
  1 - spark-nlp format.  
  2 - text. This format is usually used by [Glove](https://nlp.stanford.edu/projects/glove/)  
  3 - binary. This format is usually used by [Word2Vec](https://code.google.com/archive/p/word2vec/)
- setCaseSensitive: whether to ignore case in tokens for embeddings matching

**Example:**

{% highlight python %}
word_embeddings = WordEmbeddings() \
        .setInputCols(["document", "token"])\
        .setOutputCol("word_embeddings")
        .setEmbeddingsSource('./embeddings.100d.test.txt', 100, 2)
{% endhighlight %}

{% highlight scala %}
wordEmbeddings = new WordEmbeddings()
        .setInputCols("document", "token")
        .setOutputCol("word_embeddings")
        .setEmbeddingsSource("./embeddings.100d.test.txt",
        100, WordEmbeddingsFormat.TEXT)
{% endhighlight %}

### NER CRF

#### Named Entity Recognition CRF annotator

This Named Entity recognition annotator allows for a generic model to be trained by utilizing a CRF machine learning algorithm. Its train data (train_ner) is either a labeled or an [external CoNLL 2003 IOB based](#TrainCoNLL) spark dataset with Annotations columns. Also the user has to provide [word embeddings annotation](#WordEmbeddings) column.  
Optionally the user can provide an entity dictionary file for better accuracy  
**Type:** Named_Entity  
**Requires:** Document, Token, POS, Word_Embeddings  
**Functions:**

- setLabelColumn: If DatasetPath is not provided, this Seq\[Annotation\] type of column should have labeled data per token
- setMinEpochs: Minimum number of epochs to train
- setMaxEpochs: Maximum number of epochs to train
- setL2: L2 regularization coefficient for CRF
- setC0: c0 defines decay speed for gradient
- setLossEps: If epoch relative improvement lass than this value, training is stopped
- setMinW: Features with less weights than this value will be filtered out
- setExternalFeatures(path, delimiter, readAs, options): Path to file or folder of line separated file that has something like this: Volvo:ORG with such delimiter, readAs LINE_BY_LINE or SPARK_DATASET with options passed to the latter.
- setEntities: Array of entities to recognize
- setVerbose: Verbosity level
- setRandomSeed: Random seed

**Example:**

{% highlight python %}
nerTagger = NerCrfApproach()\
    .setInputCols(["sentence", "token", "pos"])\
    .setLabelColumn("label")\
    .setOutputCol("ner")\
    .setMinEpochs(1)\
    .setMaxEpochs(20)\
    .setLossEps(1e-3)\
    .setDicts(["ner-corpus/dict.txt"])\
    .setL2(1)\
    .setC0(1250000)\
    .setRandomSeed(0)\
    .setVerbose(2)
    .fit(train_ner)
{% endhighlight %}

{% highlight scala %}
val nerTagger = new NerCrfApproach()
    .setInputCols("sentence", "token", "pos")
    .setLabelColumn("label")
    .setMinEpochs(1)
    .setMaxEpochs(3)
    .setC0(34)
    .setL2(3.0)
    .setOutputCol("ner")
    .fit(trainNer)
{% endhighlight %}

### NER DL

#### Named Entity Recognition Deep Learning annotator

This Named Entity recognition annotator allows to train generic NER model based on Neural Networks. Its train data (train_ner) is either a labeled or an [external CoNLL 2003 IOB based](#TrainCoNLL) spark dataset with Annotations columns. Also the user has to provide [word embeddings annotation](#WordEmbeddings) column.  
Neural Network architecture is Char CNN - BLSTM that achieves state-of-the-art in most datasets.  
**Type:** Named_Entity  
**Requires:** Document, Token, Word_Embeddings  
**Functions:**

- setLabelColumn: If DatasetPath is not provided, this Seq\[Annotation\] type of column should have labeled data per token
- setMaxEpochs: Maximum number of epochs to train
- setLr: Initial learning rate
- setPo: Learning rate decay coefficient. Real Learning Rate: lr / (1 + po \* epoch)
- setBatchSize: Batch size for training
- setDropout: Dropout coefficient
- setVerbose: Verbosity level
- setRandomSeed: Random seed

**Example:**

{% highlight python %}
nerTagger = NerDLApproach()\
    .setInputCols(["sentence", "token"])\
    .setLabelColumn("label")\
    .setOutputCol("ner")\
    .setMaxEpochs(10)\
    .setRandomSeed(0)\
    .setVerbose(2)
    .fit(train_ner)
{% endhighlight %}

{% highlight scala %}
val nerTagger = new NerDLApproach()
        .setInputCols("sentence", "token")
        .setOutputCol("ner")
        .setLabelColumn("label")
        .setMaxEpochs(120)
        .setRandomSeed(0)
        .setPo(0.03f)
        .setLr(0.2f)
        .setDropout(0.5f)
        .setBatchSize(9)
        .setVerbose(Verbose.Epochs)
        .fit(trainNer)
{% endhighlight %}

### Norvig SpellChecker

This annotator retrieves tokens and makes corrections automatically if not found in an English dictionary  
**Type:** Token  
**Inputs:** Any text for corpus. A list of words for dictionary. A comma separated custom dictionary.  
**Requires:** Tokenizer  
**Train Data:** train_corpus is a spark dataset of text content  
**Functions:**

- setDictionary(path, tokenPattern, readAs, options): path to file with properly spelled words, tokenPattern is the regex pattern to identify them in text, readAs LINE_BY_LINE or SPARK_DATASET, with options passed to Spark reader if the latter is set.
- setSlangDictionary(path, delimiter, readAs, options): path to custom word mapping for spell checking. e.g. gr8 -> great. Uses provided delimiter, readAs LINE_BY_LINE or SPARK_DATASET with options passed to reader if the latter.
- setCaseSensitive(boolean): defaults to false. Might affect accuracy
- setDoubleVariants(boolean): enables extra check for word combinations, more accuracy at performance
- setShortCircuit(boolean): faster but less accurate mode
- setWordSizeIgnore(int): Minimum size of word before moving on. Defaults to 3.
- setDupsLimit(int): Maximum duplicate of characters to account for. Defaults to 2.
- setReductLimit(int): Word reduction limit. Defaults to 3
- setIntersections(int): Hamming intersections to attempt. Defaults to 10.
- setVowelSwapLimit(int): Vowel swap attempts. Defaults to 6.

**Example:**

{% highlight python %}
spell_checker = NorvigSweetingApproach() \
    .setInputCols(["token"]) \
    .setOutputCol("spell") \
    .fit(train_corpus)
{% endhighlight %}

{% highlight scala %}
val spellChecker = new NorvigSweetingApproach()
    .setInputCols(Array("normalized"))
    .setOutputCol("spell")
    .fit(trainCorpus)
{% endhighlight %}

### Symmetric SpellChecker

This spell checker is inspired on Symmetric Delete algorithm. It retrieves tokens and utilizes distance metrics to compute possible derived words  
**Type:** Token  
**Inputs:** Any text for corpus. A list of words for dictionary. A comma separated custom dictionary.  
**Requires:** Tokenizer  
**Train Data:** train_corpus is a spark dataset of text content  
**Functions:**

- setDictionary(path, tokenPattern, readAs, options): Optional dictionary of properly written words. If provided, significantly boosts spell checking performance
- setMaxEditDistance(distance): Maximum edit distance to calculate possible derived words. Defaults to 3.

**Example:**

{% highlight python %}
spell_checker = SymmetricDeleteApproach() \
    .setInputCols(["token"]) \
    .setOutputCol("spell") \
    .fit(train_corpus)
{% endhighlight %}

{% highlight scala %}
val spellChecker = new SymmetricDeleteApproach()
    .setInputCols(Array("normalized"))
    .setOutputCol("spell")
    .fit(trainCorpus)
{% endhighlight %}

### Dependency Parser

#### Unlabeled grammatical relation

Unlabeled parser that finds a grammatical relation between two words in a sentence. Its input is a directory with dependency treebank files.  
**Type:** Dependency  
**Requires:** Document, POS, Token  
**Functions:**

- setNumberOfIterations: Number of iterations in training, converges to better accuracy
- setDependencyTreeBank: Dependency treebank folder with files in [Penn Treebank format](http://www.nltk.org/nltk_data/)
- conllU: Path to a file in [CoNLL-U format](https://universaldependencies.org/format.html)

**Example:**

{% highlight python %}
dependency_parser = DependencyParserApproach() \
            .setInputCols(["sentence", "pos", "token"]) \
            .setOutputCol("dependency") \
            .setDependencyTreeBank("file://parser/dependency_treebank") \
            .setNumberOfIterations(10)
{% endhighlight %}

{% highlight scala %}
val dependencyParser = new DependencyParserApproach()
    .setInputCols(Array("sentence", "pos", "token"))
    .setOutputCol("dependency")
    .setDependencyTreeBank("parser/dependency_treebank")
    .setNumberOfIterations(10)
{% endhighlight %}

### Typed Dependency Parser

#### Labeled grammatical relation

Labeled parser that finds a grammatical relation between two words in a sentence. Its input is a CoNLL2009 or ConllU dataset.  
**Type:** Labeled Dependency  
**Requires:** Token, POS, Dependency  
**Functions:**

- setNumberOfIterations: Number of iterations in training, converges to better accuracy
- setConll2009: Path to a file in [CoNLL 2009 format](https://ufal.mff.cuni.cz/conll2009-st/trial-data.html)
- setConllU: Path to a file in [CoNLL-U format](https://universaldependencies.org/format.html)

**Example:**

{% highlight python %}
typed_dependency_parser = TypedDependencyParserApproach() \
            .setInputCols(["token", "pos", "dependency"]) \
            .setOutputCol("labdep") \
            .setConll2009("file://conll2009/eng.train") \
            .setNumberOfIterations(10)
{% endhighlight %}

{% highlight scala %}
val typedDependencyParser = new TypedDependencyParserApproach()
    .setInputCols(Array("token", "pos", "dependency"))
    .setOutputCol("labdep")
    .setConll2009("conll2009/eng.train"))
{% endhighlight %}
