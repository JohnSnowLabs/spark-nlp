---
layout: docs
header: true
title: Annotators
permalink: /docs/en/annotators
key: docs-annotators
modify_date: "2021-02-01"
use_language_switcher: "Python-Scala"
---

<div class="h3-box" markdown="1">

## How to read this section

All annotators in Spark NLP share a common interface, this is:

- Annotation -> `Annotation(annotatorType, begin, end, result, meta-data,
embeddings)`
- AnnotatorType -> some annotators share a type. This is not only
figurative, but also tells about the structure of the `metadata` map in
the Annotation. This is the one referred in the input and output of
annotators.
- Inputs -> Represents how many and which annotator types are expected
in `setInputCols`. These are column names of output of other annotators
in the dataframe.
- Output -> Represents the type of the output in the column
`setOutputCol`.

There are two types of annotators:

- Approach -> AnnotatorApproach extend Estimators, which are meant to be trained through `fit()`
- Model -> AnnotatorModel extend from Transfromers, which are meant to transform dataframes through `transform()`

`Model` suffix is explicitly stated when the annotator is the result of a training process. Some annotators, such as `Tokenizer` are transformers, but do not contain the word Model since they are not trained annotators.

`Model` annotators have a `pretrained()` on it's static object, to retrieve the public pretrained version of a model.

- pretrained(name, language, extra_location) -> by default, pretrained will bring a default model, sometimes we offer more than one model, in this case, you may have to use name, language or extra location to download them.

The types are:

- DOCUMENT = "document"
- TOKEN = "token"
- WORDPIECE = "wordpiece"
- WORD_EMBEDDINGS = "word_embeddings"
- SENTENCE_EMBEDDINGS = "sentence_embeddings"
- CATEGORY = "category"
- DATE = "date"
- ENTITY = "entity"
- SENTIMENT = "sentiment"
- POS = "pos"
- CHUNK = "chunk"
- NAMED_ENTITY = "named_entity"
- NEGEX = "negex"
- DEPENDENCY = "dependency"
- LABELED_DEPENDENCY = "labeled_dependency"
- LANGUAGE = "language"
- KEYWORD = "keyword"

{:.table-model-big}
|Annotator|Description|Version |
|---|---|---|
|Tokenizer|Identifies tokens with tokenization open standards|Opensource|
|WordSegmenter|Trainable annotator for word segmentation of languages without any rule-based tokenization such as Chinese, Japanese, or Korean|Opensource|
|Normalizer|Removes all dirty characters from text|Opensource|
|DocumentNormalizer|Cleaning content from HTML or XML documents|Opensource|
|Stemmer|Returns hard-stems out of words with the objective of retrieving the meaningful part of the word|Opensource|
|Lemmatizer|Retrieves lemmas out of words with the objective of returning a base dictionary word|Opensource|
|StopWordsCleaner|This annotator excludes from a sequence of strings (e.g. the output of a Tokenizer, Normalizer, Lemmatizer, and Stemmer) and drops all the stop words from the input sequences|Opensource|
|RegexMatcher|Uses a reference file to match a set of regular expressions and put them inside a provided key.|Opensource|
|TextMatcher|Annotator to match entire phrases (by token) provided in a file against a Document|Opensource|
|Chunker|Matches a pattern of part-of-speech tags in order to return meaningful phrases from document|Opensource|
|NGramGenerator|integrates Spark ML NGram function into Spark ML with a new cumulative feature to also generate range ngrams like the scikit-learn library|Opensource|
|DateMatcher|Reads from different forms of date and time expressions and converts them to a provided date format|Opensource|
|MultiDateMatcher|Reads from multiple different forms of date and time expressions and converts them to a provided date format|Opensource|
|SentenceDetector|Finds sentence bounds in raw text. Applies rules from Pragmatic Segmenter|Opensource|
|POSTagger|Sets a Part-Of-Speech tag to each word within a sentence. |Opensource|
|ViveknSentimentDetector|Scores a sentence for a sentiment|Opensource|
|SentimentDetector|Scores a sentence for a sentiment|Opensource|
|WordEmbeddings|Word Embeddings lookup annotator that maps tokens to vectors|Opensource|
|BertEmbeddings|BERT (Bidirectional Encoder Representations from Transformers) provides dense vector representations for natural language by using a deep, pre-trained neural network with the Transformer architecture|Opensource|
|BertSentenceEmbeddings|This annotator generates sentence embeddings from all BERT models|Opensource|
|ElmoEmbeddings|Computes contextualized word representations using character-based word representations and bidirectional LSTMs|Opensource|
|AlbertEmbeddings|Computes contextualized word representations using "A Lite" implementation of BERT algorithm by applying parameter-reduction techniques|Opensource|
|XlnetEmbeddings|Computes contextualized word representations using combination of Autoregressive Language Model and Permutation Language Model|Opensource|
|UniversalSentenceEncoder|Encodes text into high dimensional vectors that can be used for text classification, semantic similarity, clustering and other natural language tasks.|Opensource|
|SentenceEmbeddings|utilizes WordEmbeddings or BertEmbeddings to generate sentence or document embeddings|Opensource|
|ChunkEmbeddings|utilizes WordEmbeddings or BertEmbeddings to generate chunk embeddings from either Chunker, NGramGenerator, or NerConverter outputs|Opensource|
|ClassifierDL|Multi-class Text Classification. ClassifierDL uses the state-of-the-art Universal Sentence Encoder as an input for text classifications. The ClassifierDL annotator uses a deep learning model (DNNs) we have built inside TensorFlow and supports up to 100 classes|Opensource|
|MultiClassifierDL|Multi-label Text Classification. MultiClassifierDL uses a Bidirectional GRU with Convolution model that we have built inside TensorFlow and supports up to 100 classes.|Opensource|
|SentimentDL|Multi-class Sentiment Analysis Annotator. SentimentDL is an annotator for multi-class sentiment analysis. This annotator comes with 2 available pre-trained models trained on IMDB and Twitter datasets|Opensource|
|T5Transformer|for Text-To-Text Transfer Transformer (Google T5) models to achieve state-of-the-art results on multiple NLP tasks such as Translation, Summarization, Question Answering, Sentence Similarity, and so on|Opensource|
|MarianTransformer|Neural Machine Translation based on MarianNMT models being developed by the Microsoft Translator team|Opensource|
|LanguageDetectorDL|State-of-the-art language detection and identification annotator trained by using TensorFlow/keras neural networks|Opensource|
|YakeModel|Yake is an Unsupervised, Corpus-Independent, Domain and Language-Independent and Single-Document keyword extraction algorithm.|Opensource|
|NerDL|Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings)|Opensource|
|NerCrf|Named Entity recognition annotator allows for a generic model to be trained by utilizing a CRF machine learning algorithm|Opensource|
|NorvigSweeting SpellChecker|This annotator retrieves tokens and makes corrections automatically if not found in an English dictionary|Opensource|
|SymmetricDelete SpellChecker|This spell checker is inspired on Symmetric Delete algorithm|Opensource|
|Context SpellChecker|Implements Noisy Channel Model Spell Algorithm. Correction candidates are extracted combining context information and word information|Opensource|
|DependencyParser|Unlabeled parser that finds a grammatical relation between two words in a sentence|Opensource|
|TypedDependencyParser|Labeled parser that finds a grammatical relation between two words in a sentence|Opensource|
|PubTator reader|Converts automatic annotations of the biomedical datasets into Spark DataFrame|Opensource|

</div>

<div class="h3-box" markdown="1">

## Tokenizer

Identifies tokens with tokenization open standards. A few rules will help customizing it if defaults do not fit user needs.  
**Output type:** Token  
**Input types:** Document  
**Reference:** [Tokenizer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/Tokenizer.scala)|[TokenizerModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/TokenizerModel.scala)  
**Functions:**

- setExceptions(StringArray): List of tokens to not alter at all. Allows composite tokens like two worded tokens that the user may not want to split.
- addException(String): Add a single exception
- setExceptionsPath(String): Path to txt file with list of token exceptions
- caseSensitiveExceptions(bool): Whether to follow case sensitiveness for matching exceptions in text
- contextChars(StringArray): List of 1 character string to rip off from tokens, such as parenthesis or question marks. Ignored if using prefix, infix or suffix patterns.
- splitChars(StringArray): List of 1 character string to split tokens inside, such as hyphens. Ignored if using infix, prefix or suffix patterns.
- splitPattern (String): pattern to separate from the inside of tokens. takes priority over splitChars.
- setTargetPattern: Basic regex rule to identify a candidate for tokenization. Defaults to `\\S+` which means anything not a space
- setSuffixPattern: Regex to identify subtokens that are in the end of the token. Regex has to end with `\\z` and must contain groups (). Each group will become a separate token within the prefix. Defaults to non-letter characters. e.g. quotes or parenthesis
- setPrefixPattern: Regex to identify subtokens that come in the beginning of the token. Regex has to start with `\\A` and must contain groups (). Each group will become a separate token within the prefix. Defaults to non-letter characters. e.g. quotes or parenthesis
- addInfixPattern: Add an extension pattern regex with groups to the top of the rules (will target first, from more specific to the more general).
- minLength: Set the minimum allowed legth for each token
- maxLength: Set the maximum allowed legth for each token

**Note:** all these APIs receive regular expressions so please make sure that you escape special characters according to Java conventions.  

**Example:**

Refer to the [Tokenizer](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.Tokenizer) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
tokenizer = Tokenizer() \
    .setInputCols(["sentences"]) \
    .setOutputCol("token") \
    .setSplitChars(['-']) \
    .setContextChars(['(', ')', '?', '!']) \
    .addException("New York") \
    .addException("e-mail")
```

```scala
val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")
    .setContextChars(Array("(", ")", "?", "!"))
    .setSplitChars(Array('-'))
    .addException("New York")
    .addException("e-mail")
```

</div></div><div class="h3-box" markdown="1">

## DocumentNormalizer (Text cleaning)

Annotator which normalizes raw text from tagged text, e.g. scraped web pages or xml documents, from document type columns into Sentence.  
**Output type:** Document  
**Input types:** Document  
**Reference:** [DocumentNormalizer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/DocumentNormalizer.scala)  
**Functions:**

- setCleanupPatterns(patterns): normalization regex patterns which match will be removed from document. Defaults is "<[^>]*>".
- setLowercase(value): whether to convert strings to lowercase, default false
- setRemovalPolicy(policy): removalPolicy to remove pattern from text

**Example:**

Refer to the [DocumentNormalizer](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.Normalizer) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
documentNormalizer = DocumentNormalizer() \
      .setInputCols("document") \
      .setOutputCol("normalizedDocument") \
      .setCleanupPatterns(cleanUpPatterns) \
      .setRemovalPolicy(removalPolicy)
```

```scala
    val documentNormalizer = new DocumentNormalizer()
      .setInputCols("document")
      .setOutputCol("normalizedDocument")
      .setCleanupPatterns(cleanUpPatterns)
      .setRemovalPolicy(removalPolicy)
```

</div></div><div class="h3-box" markdown="1">

## Normalizer (Text cleaning)

Removes all dirty characters from text following a regex pattern and transforms words based on a provided dictionary  
**Output type:** Token  
**Input types:** Token  
**Reference:** [Normalizer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/Normalizer.scala) | [NormalizerModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/NormalizerModel.scala)  
**Functions:**

- setCleanupPatterns(patterns): Regular expressions list for normalization, defaults \[^A-Za-z\]
- setLowercase(value): lowercase tokens, default true
- setSlangDictionary(path): txt file with delimited words to be transformed into something else

**Example:**

Refer to the [Normalizer](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.Normalizer) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")
```

```scala
val normalizer = new Normalizer()
    .setInputCols(Array("token"))
    .setOutputCol("normalized")
```

</div></div><div class="h3-box" markdown="1">

## Stemmer

Returns hard-stems out of words with the objective of retrieving the meaningful part of the word  
**Output type:** Token  
**Input types:** Token  
**Reference:** [Stemmer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/Stemmer.scala)  

**Example:**

Refer to the [Stemmer](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotator.package$$Stemmer$) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
stemmer = Stemmer() \
    .setInputCols(["token"]) \
    .setOutputCol("stem")
```

```scala
val stemmer = new Stemmer()
    .setInputCols(Array("token"))
    .setOutputCol("stem")
```

</div></div><div class="h3-box" markdown="1">

## Lemmatizer

Retrieves lemmas out of words with the objective of returning a base dictionary word  
**Output type:** Token  
**Input types:** Token  
**Input:** abduct -> abducted abducting abduct abducts  
**Reference:** [Lemmatizer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/Lemmatizer.scala) | [LemmatizerModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/LemmatizerModel.scala)  
**Functions:** 

- setDictionary(path, keyDelimiter, valueDelimiter, readAs, options): Path and options to lemma dictionary, in lemma vs possible words format. readAs can be LINE_BY_LINE or SPARK_DATASET. options contain option passed to spark reader if readAs is SPARK_DATASET.

**Example:**

Refer to the [Lemmatizer](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.Lemmatizer) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
lemmatizer = Lemmatizer() \
    .setInputCols(["token"]) \
    .setOutputCol("lemma") \
    .setDictionary("./lemmas001.txt")
```

```scala
val lemmatizer = new Lemmatizer()
    .setInputCols(Array("token"))
    .setOutputCol("lemma")
    .setDictionary("./lemmas001.txt")
```

</div></div><div class="h3-box" markdown="1">

## StopWordsCleaner

This annotator excludes from a sequence of strings (e.g. the output of a `Tokenizer`, `Normalizer`, `Lemmatizer`, and `Stemmer`) and drops all the stop words from the input sequences.

**Functions:**

- `setStopWords`: The words to be filtered out. `Array[String]`
- `setCaseSensitive`: Whether to do a case sensitive comparison over the stop words.

**Example:**

Refer to the [StopWordsCleaner](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.StopWordsCleaner) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
stop_words_cleaner = StopWordsCleaner() \
        .setInputCols(["token"]) \
        .setOutputCol("cleanTokens") \
        .setCaseSensitive(False) \
        .setStopWords(["this", "is", "and"])
```

```scala
val stopWordsCleaner = new StopWordsCleaner()
      .setInputCols("token")
      .setOutputCol("cleanTokens")
      .setStopWords(Array("this", "is", "and"))
      .setCaseSensitive(false)
```



**NOTE:**
If you need to `setStopWords` from a text file, you can first read and convert it into an array of string:

```python
# your stop words text file, each line is one stop word
stopwords = sc.textFile("/tmp/stopwords/english.txt").collect()
# simply use it in StopWordsCleaner
stopWordsCleaner = new StopWordsCleaner()
      .setInputCols("token")
      .setOutputCol("cleanTokens")
      .setStopWords(stopwords)
      .setCaseSensitive(false)
```

```scala
// your stop words text file, each line is one stop word
val stopwords = sc.textFile("/tmp/stopwords/english.txt").collect()
// simply use it in StopWordsCleaner
val stopWordsCleaner = new StopWordsCleaner()
      .setInputCols("token")
      .setOutputCol("cleanTokens")
      .setStopWords(stopwords)
      .setCaseSensitive(false)
```

</div></div><div class="h3-box" markdown="1">

## RegexMatcher

Uses a reference file to match a set of regular expressions and put them inside a provided key. File must be comma separated.  
**Output type:** Regex  
**Input types:** Document  
**Input:** `the\\s\\w+`, "followed by 'the'"  
**Reference:** [RegexMatcher](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/RegexMatcher.scala) | [RegexMatcherModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/RegexMatcherModel.scala)  
**Functions:**

- setStrategy(strategy): Can be any of `MATCH_FIRST|MATCH_ALL|MATCH_COMPLETE`
- setRules(path, delimiter, readAs, options): Path to file containing a set of regex,key pair. readAs can be LINE_BY_LINE or SPARK_DATASET. options contain option passed to spark reader if readAs is SPARK_DATASET.

**Example:**

Refer to the [RegexMatcher](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.RegexMatcher) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
regex_matcher = RegexMatcher() \
    .setStrategy("MATCH_ALL") \
    .setInputCols("document")
    .setOutputCol("regex")
```

```scala
val regexMatcher = new RegexMatcher()
    .setStrategy("MATCH_ALL")
    .setInputCols(Array("document"))
    .setOutputCol("regex")
```

</div></div><div class="h3-box" markdown="1">

## TextMatcher (Phrase matching)

Annotator to match entire phrases (by token) provided in a file against a Document  
**Output type:** Entity  
**Input types:** Document, Token  
**Input:** hello world, I am looking for you  
**Reference:** [TextMatcher](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/TextMatcher.scala) | [TextMatcherModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/TextMatcherModel.scala)  
**Functions:**

- setEntities(path, format, options): Provides a file with phrases to match. Default: Looks up path in configuration.  
- path: a path to a file that contains the entities in the specified format.  
- readAs: the format of the file, can be one of {ReadAs.LINE_BY_LINE, ReadAs.SPARK_DATASET}. Defaults to LINE_BY_LINE.  
- options: a map of additional parameters. Defaults to {"format": "text"}.

**Example:**

Refer to the [TextMatcher](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.TextMatcher) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
entity_extractor = TextMatcher() \
    .setInputCols(["inputCol"])\
    .setOutputCol("entity")\
    .setEntities("/path/to/file/myentities.txt")
```

```scala
val entityExtractor = new TextMatcher()
    .setInputCols("inputCol")
    .setOutputCol("entity")
    .setEntities("/path/to/file/myentities.txt")
```

</div></div><div class="h3-box" markdown="1">

## Chunker

This annotator matches a pattern of part-of-speech tags in order to return meaningful phrases from document

**Output type:** Chunk  
**Input types:** Document, POS  
**Reference:** [Chunker](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/Chunker.scala)  
**Functions:**

- setRegexParsers(patterns): A list of regex patterns to match chunks, for example: Array("‹DT›?‹JJ›\*‹NN›")
- addRegexParser(patterns): adds a pattern to the current list of chunk patterns, for example: "‹DT›?‹JJ›\*‹NN›"

**Example:**

Refer to the [Chunker](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.Chunker) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
chunker = Chunker() \
    .setInputCols(["document", "pos"]) \
    .setOutputCol("chunk") \
    .setRegexParsers(["‹NNP›+", "‹DT|PP\\$›?‹JJ›*‹NN›"])
```

```scala
val chunker = new Chunker()
    .setInputCols(Array("document", "pos"))
    .setOutputCol("chunk")
    .setRegexParsers(Array("‹NNP›+", "‹DT|PP\\$›?‹JJ›*‹NN›"))
```

</div></div><div class="h3-box" markdown="1">

## NGramGenerator

`NGramGenerator` annotator takes as input a sequence of strings (e.g. the output of a `Tokenizer`, `Normalizer`, `Stemmer`, `Lemmatizer`, and `StopWordsCleaner`). The parameter `n` is used to determine the number of terms in each n-gram. The output will consist of a sequence of n-grams where each n-gram is represented by a space-delimited string of n consecutive words with annotatorType `CHUNK` same as the `Chunker` annotator.

**Output type:** CHUNK  
**Input types:** TOKEN  
**Reference:** [NGramGenerator](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/NGramGenerator.scala)  
**Functions:**

- setN: number elements per n-gram (>=1)
- setEnableCumulative: whether to calculate just the actual n-grams or all n-grams from 1 through n
- setDelimiter: Glue character used to join the tokens

**Example:**

Refer to the [NGramGenerator](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.NGramGenerator) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
ngrams_cum = NGramGenerator() \
            .setInputCols(["token"]) \
            .setOutputCol("ngrams") \
            .setN(2) \
            .setEnableCumulative(True)
            .setDelimiter("_") # Default is space
```

```scala
val nGrams = new NGramGenerator()
      .setInputCols("token")
      .setOutputCol("ngrams")
      .setN(2)
      .setEnableCumulative(true)
      .setDelimiter("_") // Default is space
```

</div></div><div class="h3-box" markdown="1">

## DateMatcher

Reads from different forms of date and time expressions and converts them to a provided date format. Extracts only ONE date per sentence. Use with sentence detector for more matches.  
**Output type:** Date  
**Input types:** Document  
**Reference:** [DateMatcher](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/DateMatcher.scala)  
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

- setDateFormat(format): SimpleDateFormat standard date *output* formatting. Defaults to yyyy/MM/dd
- setAnchorDateYear: Add an anchor year for the relative dates such as a day after tomorrow. If not set it will use the current year. Example: 2021
- setAnchorDateMonth: Add an anchor month for the relative dates such as a day after tomorrow. If not set it will use the current month. Example: 1 which means January
- setAnchorDateDay: Add an anchor day of the day for the relative dates such as a day after tomorrow. If not set it will use the current day. Example: 11

**Example:**

Refer to the [DateMatcher](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.DateMatcher) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
date_matcher = DateMatcher() \
    .setInputCols('document')\
    .setOutputCol("date") \
    .setDateFormat("yyyy/MM/dd")
```

```scala
val dateMatcher = new DateMatcher()
    .setInputCols("document")
    .setOutputCol("date")
    .setFormat("yyyyMM")
```

## MultiDateMatcher

Reads from multiple different forms of date and time expressions and converts them to a provided date format. Extracts multiple dates per sentence.
**Output type:** Date  
**Input types:** Document  
**Reference:** [MultiDateMatcher](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/MultiDateMatcher.scala)  
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

- setDateFormat(format): SimpleDateFormat standard date *output* formatting. Defaults to yyyy/MM/dd
- setAnchorDateYear: Add an anchor year for the relative dates such as a day after tomorrow. If not set it will use the current year. Example: 2021
- setAnchorDateMonth: Add an anchor month for the relative dates such as a day after tomorrow. If not set it will use the current month. Example: 1 which means January
- setAnchorDateDay: Add an anchor day of the day for the relative dates such as a day after tomorrow. If not set it will use the current day. Example: 11

**Example:**

Refer to the [MultiDateMatcher](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.MultiDateMatcher) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
date_matcher = MultiDateMatcher() \
    .setInputCols('document')\
    .setOutputCol("date") \
    .setDateFormat("yyyy/MM/dd")
```

```scala
val dateMatcher = new MultiDateMatcher()
    .setInputCols("document")
    .setOutputCol("date")
    .setFormat("yyyyMM")
```

</div></div><div class="h3-box" markdown="1">

## SentenceDetector

Finds sentence bounds in raw text. Applies rules from Pragmatic Segmenter.  
**Output type:** Sentence
**Input types:** Document  
**Reference:** [SentenceDetector](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/sbd/pragmatic/SentenceDetector.scala)  
**Functions:**

- setCustomBounds(string): Custom sentence separator text
- setUseCustomOnly(bool): Use only custom bounds without considering those of Pragmatic Segmenter. Defaults to false. Needs customBounds.
- setUseAbbreviations(bool): Whether to consider abbreviation strategies for better accuracy but slower performance. Defaults to true.
- setExplodeSentences(bool): Whether to split sentences into different Dataset rows. Useful for higher parallelism in fat rows. Defaults to false.

**Example:**

Refer to the [SentenceDetector](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
sentence_detector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")
```

```scala
val sentenceDetector = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentence")
```

</div></div><div class="h3-box" markdown="1">

## POSTagger (Part of speech tagger)

Sets a POS tag to each word within a sentence. Its train data (train_pos) is a spark dataset of [POS format values](#TrainPOS) with Annotation columns.  
**Output type:** POS  
**Input types:** Document, Token  
**Reference:** [PerceptronApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/pos/perceptron/PerceptronApproach.scala) | [PerceptronModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/pos/perceptron/PerceptronModel.scala)  
**Functions:**

- setNIterations(number): Number of iterations for training. May improve accuracy but takes longer. Default 5.
- setPosColumn(colname): Column containing an array of POS Tags matching every token on the line.

**Example:**

Refer to the [PerceptronApproach](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronApproach) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
pos_tagger = PerceptronApproach() \
    .setInputCols(["token", "sentence"]) \
    .setOutputCol("pos") \
    .setNIterations(2) \
    .fit(train_pos)
```

```scala
val posTagger = new PerceptronApproach()
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("pos")
    .setNIterations(2)
    .fit(trainPOS)
```

</div></div><div class="h3-box" markdown="1">

## ViveknSentimentDetector

Scores a sentence for a sentiment
  
**Output type:** sentiment  
**Input types:** Document, Token  
**Reference:** [ViveknSentimentApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/sda/vivekn/ViveknSentimentApproach.scala) | [ViveknSentimentModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/sda/vivekn/ViveknSentimentModel.scala)  
**Functions:**

- setSentimentCol(colname): Column with sentiment analysis row's result for training. If not set, external sources need to be set instead.
- setSentimentCol(colname): column with the sentiment result of every row. Must be 'positive' or 'negative'
- setCorpusPrune(true): when training on small data you may want to disable this to not cut off infrequent words

**Input:** File or folder of text files of positive and negative data  
**Example:**

Refer to the [ViveknSentimentApproach](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentApproach) Scala docs for more details on the API.

Train your own model:

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
sentiment_detector = ViveknSentimentApproach() \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("sentiment") \
    .setSentimentCol("sentiment_label")
```

```scala
val sentimentDetector = new ViveknSentimentApproach()
      .setInputCols(Array("token", "sentence"))
      .setOutputCol("vivekn")
      .setSentimentCol("sentiment_label")
      .setCorpusPrune(0)
```

</div>

Use a pretrained model:

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
sentiment_detector = ViveknSentimentModel.pretrained() \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("sentiment")
```

```scala
val sentimentDetector = new ViveknSentimentModel.pretrained
      .setInputCols(Array("token", "sentence"))
      .setOutputCol("vivekn")
```


</div></div><div class="h3-box" markdown="1">

## SentimentDetector (Sentiment analysis)

Scores a sentence for a sentiment  
**Output type:** sentiment  

**Input types:** Document, Token  

**Reference:** [SentimentDetector](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/sda/pragmatic/SentimentDetector.scala) | [SentimentDetectorModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/sda/pragmatic/SentimentDetectorModel.scala)  
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

Refer to the [SentimentDetector](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetector) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
sentiment_detector = SentimentDetector() \
    .setInputCols(["token", "sentence"]) \
    .setOutputCol("sentiment")
```

```scala
val sentimentDetector = new SentimentDetector
    .setInputCols(Array("token", "sentence"))
    .setOutputCol("sentiment")
```

</div></div><div class="h3-box" markdown="1">

## WordEmbeddings

Word Embeddings lookup annotator that maps tokens to vectors  

**Output type:** Word_Embeddings  

**Input types:** Document, Token  

**Reference:**  [WordEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/WordEmbeddings.scala) | [WordEmbeddingsModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/WordEmbeddingsModel.scala)  
**Functions:**

- setStoragePath(path, format): sets [word embeddings](https://en.wikipedia.org/wiki/Word_embedding) options. 
  - path: word embeddings file  
  - format: format of word embeddings files:
    - TEXT -> This format is usually used by [Glove](https://nlp.stanford.edu/projects/glove/)
    - BINARY -> This format is usually used by [Word2Vec](https://code.google.com/archive/p/word2vec/)
- setCaseSensitive: whether to ignore case in tokens for embeddings matching

**Example:**

Refer to the [WordEmbeddings](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.embeddings.WordEmbeddings) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
embeddings = WordEmbeddings()
      .setStoragePath("/tmp/glove.6B.100d.txt", "TEXT")\
      .setDimension(100)\
      .setStorageRef("glove_100d") \
      .setInputCols("document", "token") \
      .setOutputCol("embeddings")
```

```scala
val embeddings = new WordEmbeddings()
      .setStoragePath("/tmp/glove.6B.100d.txt", "TEXT)
      .setDimension(100)
      .setStorageRef("glove_100d") // Use or save this WordEmbeddings with storageRef
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
```

</div>

There are also two convenient functions to retrieve the embeddings coverage with respect to the transformed dataset:  

- withCoverageColumn(dataset, embeddingsCol, outputCol): Adds a custom column with **word coverage** stats for the embedded field: (coveredWords, totalWords, coveragePercentage). This creates a new column with statistics for each row.
- overallCoverage(dataset, embeddingsCol): Calculates overall **word coverage** for the whole data in the embedded field. This returns a single coverage object considering all rows in the field.

</div><div class="h3-box" markdown="1">

## BertEmbeddings

BERT (Bidirectional Encoder Representations from Transformers) provides dense vector representations for natural language by using a deep, pre-trained neural network with the Transformer architecture

You can find the pre-trained models for `BertEmbeddings` in the [Spark NLP Models](https://github.com/JohnSnowLabs/spark-nlp-models) repository

**Output type:** Word_Embeddings  

**Input types:** Document, Token

**Reference:** [BertEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/BertEmbeddings.scala)  

Refer to the [BertEmbeddings](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.embeddings.BertEmbeddings) Scala docs for more

How to use pretrained BertEmbeddings:

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python

bert = BertEmbeddings.pretrained() \
      .setInputCols("sentence", "token") \
      .setOutputCol("bert")
```

```scala
val bert = BertEmbeddings.pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("bert")
```

</div><div class="h3-box" markdown="1">

## BertSentenceEmbeddings

BERT (Bidirectional Encoder Representations from Transformers) provides dense vector representations for natural language by using a deep, pre-trained neural network with the Transformer architecture

You can find the pre-trained models for `BertEmbeddings` in the [Spark NLP Models](https://github.com/JohnSnowLabs/spark-nlp-models) repository

**Output type:** Sentence_Embeddings  

**Input types:** Document

**Reference:** [BertSentenceEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/BertSentenceEmbeddings.scala)  

Refer to the [BertSentenceEmbeddings](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.embeddings.BertSentenceEmbeddings) Scala docs for more

How to use pretrained BertEmbeddings:

{% include programmingLanguageSelectScalaPython.html %}

```python

bert = BertSentencembeddings.pretrained() \
      .setInputCols("document") \
      .setOutputCol("bert_sentence_embeddings")
```

```scala
val bert = BertEmbeddings.pretrained()
      .setInputCols("document")
      .setOutputCol("bert_sentence_embeddings")
```

</div></div><div class="h3-box" markdown="1">

## ElmoEmbeddings

Computes contextualized word representations using character-based word representations and bidirectional LSTMs

You can find the pre-trained model for `ElmoEmbeddings` in the  [Spark NLP Models](https://github.com/JohnSnowLabs/spark-nlp-models#english---models) repository

**Output type:** Word_Embeddings

**Input types:** Document, Token

**Reference:** [ElmoEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/ElmoEmbeddings.scala)  

Refer to the [ElmoEmbeddings](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.embeddings.ElmoEmbeddings) Scala docs for more

How to use pretrained ElmoEmbeddings:

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
# Online - Download the pretrained model
elmo = ElmoEmbeddings.pretrained()
      .setInputCols("sentence", "token") \
      .setOutputCol("elmo")

# Offline - Download the pretrained model manually and extract it
elmo = ElmoEmbeddings.load("/elmo_en_2.4.0_2.4_1580488815299") \
        .setInputCols("sentence", "token") \
        .setOutputCol("elmo")
```

```scala

val elmo = ElmoEmbeddings.pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("elmo")
      .setPoolingLayer("elmo") //  word_emb, lstm_outputs1, lstm_outputs2 or elmo
```

</div></div><div class="h3-box" markdown="1">

## AlbertEmbeddings

Computes contextualized word representations using "A Lite" implementation of BERT algorithm by applying parameter-reduction techniques

You can find the pre-trained model for `AlbertEmbeddings` in the  [Spark NLP Models](https://github.com/JohnSnowLabs/spark-nlp-models#english---models) repository

**Functions:**

- `setBatchSize(int)`: Batch size. Large values allows faster processing but requires more memory.
- `setMaxSentenceLength(int)`: Max sentence length to process

**Output type:** Word_Embeddings

**Input types:** Document, Token

**Reference:** [AlbertEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/AlbertEmbeddings.scala)  

Refer to the [AlbertEmbeddings](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.embeddings.AlbertEmbeddings) Scala docs for more

How to use pretrained AlbertEmbeddings:

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
# Online - Download the pretrained model
albert = AlbertEmbeddings.pretrained()
      .setInputCols("sentence", "token") \
      .setOutputCol("albert")

# Offline - Download the pretrained model manually and extract it
albert = AlbertEmbeddings.load("/albert_base_uncased_en_2.5.0_2.4_1588073363475") \
        .setInputCols("sentence", "token") \
        .setOutputCol("albert")
```

```scala

val albert = AlbertEmbeddings.pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("albert")
```

</div></div><div class="h3-box" markdown="1">

## XlnetEmbeddings

Computes contextualized word representations using combination of Autoregressive Language Model and Permutation Language Model

You can find the pre-trained model for `XlnetEmbeddings` in the  [Spark NLP Models](https://github.com/JohnSnowLabs/spark-nlp-models#english---models) repository

**Functions:**

- `setBatchSize(int)`: Batch size. Large values allows faster processing but requires more memory.
- `setMaxSentenceLength(int)`: Max sentence length to process.

**Output type:** Word_Embeddings

**Input types:** Document, Token

**Reference:** [XlnetEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/XlnetEmbeddings.scala)  

Refer to the [XlnetEmbeddings](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.embeddings.XlnetEmbeddings) Scala docs for more

How to use pretrained XlnetEmbeddings:

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
# Online - Download the pretrained model
xlnet = XlnetEmbeddings.pretrained()
      .setInputCols("sentence", "token") \
      .setOutputCol("xlnet")

# Offline - Download the pretrained model manually and extract it
xlnet = XlnetEmbeddings.load("/xlnet_large_cased_en_2.5.0_2.4_1588074397954") \
        .setInputCols("sentence", "token") \
        .setOutputCol("xlnet")
```

```scala

val xlnet = XlnetEmbeddings.pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("xlnet")
```

</div></div><div class="h3-box" markdown="1">

## UniversalSentenceEncoder

The Universal Sentence Encoder encodes text into high dimensional vectors that can be used for text classification, semantic similarity, clustering and other natural language tasks.

**Output type:** SENTENCE_EMBEDDINGS

**Input types:** Document

Refer to the [UniversalSentenceEncoder](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder) Scala docs for more

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
use = UniversalSentenceEncoder.pretrained() \
            .setInputCols("sentence") \
            .setOutputCol("use_embeddings")
```

```scala
val use = new UniversalSentenceEncoder()
      .setInputCols("document")
      .setOutputCol("use_embeddings")
```

</div></div><div class="h3-box" markdown="1">

## SentenceEmbeddings

This annotator converts the results from `WordEmbeddings`, `BertEmbeddings`, `ElmoEmbeddings`, `AlbertEmbeddings`, or `XlnetEmbeddings` into `sentence` or `document` embeddings by either summing up or averaging all the word embeddings in a sentence or a document (depending on the `inputCols`).

**Functions:**

- `setPoolingStrategy`: Choose how you would like to aggregate Word Embeddings to Sentence Embeddings: AVERAGE or SUM

**Output type:** SENTENCE_EMBEDDINGS

**Input types:** Document

Refer to the [SentenceEmbeddings](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.embeddings.SentenceEmbeddings) Scala docs for more

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
sentence_embeddings = SentenceEmbeddings() \
            .setInputCols(["document", "embeddings"]) \
            .setOutputCol("sentence_embeddings") \
            .setPoolingStrategy("AVERAGE")
```

```scala
val embeddingsSentence = new SentenceEmbeddings()
      .setInputCols(Array("document", "embeddings"))
      .setOutputCol("sentence_embeddings")
      .setPoolingStrategy("AVERAGE")
```

**NOTE:**

If you choose `document` as your input for `Tokenizer`, `WordEmbeddings/BertEmbeddings`, and `SentenceEmbeddings` then it averages/sums all the embeddings into one array of embeddings. However, if you choose `sentence` as `inputCols` then for each sentence `SentenceEmbeddings` generates one array of embeddings.

**TIP:**

How to explode and convert these embeddings into `Vectors` or what's known as `Feature` column so it can be used in Spark ML regression or clustering functions:

```python
from org.apache.spark.ml.linal import Vector, Vectors
from pyspark.sql.functions import udf
# Let's create a UDF to take array of embeddings and output Vectors
@udf(Vector)
def convertToVectorUDF(matrix):
    return Vectors.dense(matrix.toArray.map(_.toDouble))


# Now let's explode the sentence_embeddings column and have a new feature column for Spark ML
pipelineDF.select(explode("sentence_embeddings.embeddings").as("sentence_embedding"))
.withColumn("features", convertToVectorUDF("sentence_embedding"))
```

```scala
import org.apache.spark.ml.linalg.{Vector, Vectors}

// Let's create a UDF to take array of embeddings and output Vectors
val convertToVectorUDF = udf((matrix : Seq[Float]) => {
    Vectors.dense(matrix.toArray.map(_.toDouble))
})

// Now let's explode the sentence_embeddings column and have a new feature column for Spark ML
pipelineDF.select(explode($"sentence_embeddings.embeddings").as("sentence_embedding"))
.withColumn("features", convertToVectorUDF($"sentence_embedding"))
```

</div></div><div class="h3-box" markdown="1">

## ChunkEmbeddings

This annotator utilizes `WordEmbeddings` or `BertEmbeddings` to generate chunk embeddings from either `Chunker`, `NGramGenerator`, or `NerConverter` outputs.

**Functions:**

- `setPoolingStrategy`: Choose how you would like to aggregate Word Embeddings to Sentence Embeddings: AVERAGE or SUM

**Output type:** CHUNK

**Input types:** CHUNK, Word_Embeddings

Refer to the [ChunkEmbeddings](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.embeddings.ChunkEmbeddings) Scala docs for more

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
chunk_embeddings = ChunkEmbeddings() \
            .setInputCols(["chunk", "embeddings"]) \
            .setOutputCol("chunk_embeddings") \
            .setPoolingStrategy("AVERAGE")
```

```scala
val chunkSentence = new ChunkEmbeddings()
      .setInputCols(Array("chunk", "embeddings"))
      .setOutputCol("chunk_embeddings")
      .setPoolingStrategy("AVERAGE")
```

**TIP:**

How to explode and convert these embeddings into `Vectors` or what's known as `Feature` column so it can be used in Spark ML regression or clustering functions:

```python
from org.apache.spark.ml.linal import Vector, Vectors
from pyspark.sql.functions import udf

// Let's create a UDF to take array of embeddings and output Vectors
@udf(Vector)
def convertToVectorUDF(matrix):
    return Vectors.dense(matrix.toArray.map(_.toDouble))

```

```scala
import org.apache.spark.ml.linalg.{Vector, Vectors}

// Let's create a UDF to take array of embeddings and output Vectors
val convertToVectorUDF = udf((matrix : Seq[Float]) => {
    Vectors.dense(matrix.toArray.map(_.toDouble))
})

// Now let's explode the sentence_embeddings column and have a new feature column for Spark ML
pipelineDF.select(explode($"chunk_embeddings.embeddings").as("chunk_embeddings_exploded"))
.withColumn("features", convertToVectorUDF($"chunk_embeddings_exploded"))
```

</div></div><div class="h3-box" markdown="1">

## ClassifierDL (Multi-class Text Classification)

ClassifierDL is a generic Multi-class Text Classification. ClassifierDL uses the state-of-the-art Universal Sentence Encoder as an input for text classifications. The ClassifierDL annotator uses a deep learning model (DNNs) we have built inside TensorFlow and supports up to 100 classes

**NOTE**: This annotator accepts a label column of a single item in either type of String, Int, Float, or Double.

**NOTE**: UniversalSentenceEncoder, BertSentenceEmbeddings, or SentenceEmbeddings can be used for the inputCol

**Output type:** CATEGORY

**Input types:** SENTENCE_EMBEDDINGS

**Functions:**

- setLabelColumn: If DatasetPath is not provided, this Seq\[Annotation\] type of column should have labeled data per token.
- setLr: Initial learning rate.
- setBatchSize: Batch size for training.
- setDropout: Dropout coefficient.
- setMaxEpochs: Maximum number of epochs to train.
- setEnableOutputLogs: Whether to output to annotators log folder.
- setValidationSplit: Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.
- setVerbose: Level of verbosity during training.
- setOutputLogsPath: Folder path to save training logs.

Refer to the [ClassifierDLApproach](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLApproach) Scala docs for more

Refer to the [ClassifierDLModel](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLModel) Scala docs for more

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
docClassifier = ClassifierDLApproach()\
      .setInputCols("sentence_embeddings")\
      .setOutputCol("category")\
      .setLabelColumn("label")\
      .setBatchSize(64)\
      .setMaxEpochs(20)\
      .setLr(0.5)\
      .setDropout(0.5)
```

```scala
val docClassifier = new ClassifierDLApproach()
      .setInputCols("sentence_embeddings")
      .setOutputCol("category")
      .setLabelColumn("label")
      .setBatchSize(64)
      .setMaxEpochs(20)
      .setLr(5e-3f)
      .setDropout(0.5f)
```

</div>

Please refer to [existing notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/jupyter/training/english/classification) for more examples.

</div><div class="h3-box" markdown="1">

## MultiClassifierDL (Multi-label Text Classification)

 MultiClassifierDL is a Multi-label Text Classification. MultiClassifierDL uses a Bidirectional GRU with Convolution model that we have built inside TensorFlow and supports up to 100 classes. The input to MultiClassifierDL is Sentence Embeddings such as state-of-the-art UniversalSentenceEncoder, BertSentenceEmbeddings, or SentenceEmbeddings

**NOTE**: This annotator accepts a label column of a single item in either type of String, Int, Float, or Double.

**NOTE**: UniversalSentenceEncoder, BertSentenceEmbeddings, or SentenceEmbeddings can be used for the inputCol

**Output type:** CATEGORY

**Input types:** SENTENCE_EMBEDDINGS

**Functions:**

- setLabelColumn: If DatasetPath is not provided, this Seq\[Annotation\] type of column should have labeled data per token.
- setLr: Initial learning rate.
- setBatchSize: Batch size for training.
- setMaxEpochs: Maximum number of epochs to train.
- setEnableOutputLogs: Whether to output to annotators log folder.
- setValidationSplit: Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.
- setVerbose: Level of verbosity during training.
- setOutputLogsPath: Folder path to save training logs.

Refer to the [MultiClassifierDLApproach](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.classifier.dl.MultiClassifierDLApproach) Scala docs for more

Refer to the [MultiClassifierDLModel](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.classifier.dl.MultiClassifierDLModel) Scala docs for more

{% include programmingLanguageSelectScalaPython.html %}

```python
docMultiClassifier = MultiClassifierDLApproach()\
      .setInputCols("sentence_embeddings")\
      .setOutputCol("category")\
      .setLabelColumn("label")\
      .setBatchSize(64)\
      .setMaxEpochs(20)\
      .setLr(0.5)
```

```scala
val docMultiClassifier = new MultiClassifierDLApproach()
      .setInputCols("sentence_embeddings")
      .setOutputCol("category")
      .setLabelColumn("label")
      .setBatchSize(64)
      .setMaxEpochs(20)
      .setLr(5e-3f)
```

Please refer to [existing notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/jupyter/training/english/classification) for more examples.

</div><div class="h3-box" markdown="1">

## SentimentDL (Multi-class Sentiment Analysis annotator)

SentimentDL is an annotator for multi-class sentiment analysis. This annotator comes with 2 available pre-trained models trained on IMDB and Twitter datasets

**NOTE**: This annotator accepts a label column of a single item in either type of String, Int, Float, or Double.

**NOTE**: UniversalSentenceEncoder, BertSentenceEmbeddings, or SentenceEmbeddings can be used for the inputCol

**Output type:** CATEGORY

**Input types:** SENTENCE_EMBEDDINGS

**Functions:**

- setLabelColumn: If DatasetPath is not provided, this Seq\[Annotation\] type of column should have labeled data per token.
- setLr: Initial learning rate.
- setBatchSize: Batch size for training.
- setDropout: Dropout coefficient.
- setThreshold: The minimum threshold for the final result otheriwse it will be either neutral or the value set in thresholdLabel.
- setThresholdLabel: In case the score is less than threshold, what should be the label. Default is neutral.
- setMaxEpochs: Maximum number of epochs to train.
- setEnableOutputLogs: Whether to output to annotators log folder.
- setOutputLogsPath: Folder path to save training logs.
- setValidationSplit: Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.
- setVerbose: Level of verbosity during training.

Refer to the [SentimentDLApproach](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.classifier.dl.SentimentDLApproach) Scala docs for more

Refer to the [SentimentDLModel](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.classifier.dl.SentimentDLModel) Scala docs for more

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
sentimentClassifier = SentimentDLApproach()\
      .setInputCols("sentence_embeddings")\
      .setOutputCol("category")\
      .setLabelColumn("label")\
      .setBatchSize(64)\
      .setMaxEpochs(20)\
      .setLr(0.5)\
      .setDropout(0.5)
```

```scala
val sentimentClassifier = new SentimentDLApproach()
      .setInputCols("sentence_embeddings")
      .setOutputCol("category")
      .setLabelColumn("label")
      .setBatchSize(64)
      .setMaxEpochs(20)
      .setLr(5e-3f)
      .setDropout(0.5f)
```

</div>

Please refer to [existing notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/classification/) for more examples.

</div><div class="h3-box" markdown="1">

## LanguageDetectorDL (Language Detection and Identiffication)

LanguageDetectorDL is a state-of-the-art language detection and identification annotator trained by using TensorFlow/keras neural networks.

**Output type:** LANGUAGE

**Input types:** DOCUMENT or SENTENCE

**Functions:**

- setThreshold: The minimum threshold for the final result otheriwse it will be either neutral or the value set in thresholdLabel.
- setThresholdLabel: In case the score is less than threshold, what should be the label. Default is Unknown.
- setCoalesceSentences: If sets to true the output of all sentences will be averaged to one output instead of one output per sentence. Default to true.

Refer to the [LanguageDetectorDL](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.ld.dl.LanguageDetectorDL) Scala docs for more

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
languageDetector = LanguageDetectorDL.pretrained("ld_wiki_20")
      .setInputCols("document")\
      .setOutputCol("language")\
      .setThreshold(0.3)\
      .setCoalesceSentences(True)
```

```scala
 val languageDetector = LanguageDetectorDL.pretrained("ld_wiki_20")
      .setInputCols("document")
      .setOutputCol("language")
      .setThreshold(0.3f)
      .setCoalesceSentences(true)
```

</div><div class="h3-box" markdown="1">

## YakeModel (Keywords Extraction)

Yake is an Unsupervised, Corpus-Independent, Domain and Language-Independent and Single-Document keyword extraction algorithm.

sExtracting keywords from texts has become a challenge for individuals and organizations as the information grows in complexity and size. The need to automate this task so that text can be processed in a timely and adequate manner has led to the emergence of automatic keyword extraction tools. Yake is a novel feature-based system for multi-lingual keyword extraction, which supports texts of different sizes, domain or languages. Unlike other approaches, Yake does not rely on dictionaries nor thesauri, neither is trained against any corpora. Instead, it follows an unsupervised approach which builds upon features extracted from the text, making it thus applicable to documents written in different languages without the need for further knowledge. This can be beneficial for a large number of tasks and a plethora of situations where access to training corpora is either limited or restricted.

The algorithm makes use of the position of a sentence and token. Therefore, to use the annotator, the text should be first sent through a Sentence Boundary Detector and then a tokenizer.

You can tweak the following parameters to get the best result from the annotator.

**Output type:** KEYWORD

**Input types:** TOKEN

**Functions:**

- setMinNGrams(int) Select the minimum length of a extracted keyword
- setMaxNGrams(int) Select the maximum length of a extracted keyword
- setNKeywords(int) Extract the top N keywords
- setStopWords(list) Set the list of stop words
- setThreshold(float) Each keyword will be given a keyword score greater than 0. (Lower the score better the keyword) Set an upper bound for the keyword score from this method.
- setWindowSize(int) Yake will construct a co-occurence matrix. You can set the window size for the cooccurence matrix construction from this method. ex: windowSize=2 will look at two words to both left and right of a candidate word.

Refer to the [YakeModel](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.keyword.yake.YakeModel) Scala docs for more

{% include programmingLanguageSelectScalaPython.html %}

```python
keywords = YakeModel() \
    .setInputCols("token") \
    .setOutputCol("keywords") \
    .setMinNGrams(1) \
    .setMaxNGrams(3)\
    .setNKeywords(20)\
    .setStopWords(stopwords)
```

```scala
 val keywords = new YakeModel()
    .setInputCols("token")
    .setOutputCol("keywords")
    .setMinNGrams(1)
    .setMaxNGrams(3)
    .setNKeywords(20)
    .setStopWords(stopwords)
```

</div><div class="h3-box" markdown="1">

## NER CRF (Named Entity Recognition CRF annotator)

This Named Entity recognition annotator allows for a generic model to be trained by utilizing a CRF machine learning algorithm. Its train data (train_ner) is either a labeled or an [external CoNLL 2003 IOB based](#conll-dataset) spark dataset with Annotations columns. Also the user has to provide [word embeddings annotation](#WordEmbeddings) column.  
Optionally the user can provide an entity dictionary file for better accuracy  
**Output type:** Named_Entity  
**Input types:** Document, Token, POS, Word_Embeddings  
**Reference:** [NerCrfApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ner/crf/NerCrfApproach.scala) | [NerCrfModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ner/crf/NerCrfModel.scala)  
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

Refer to the [NerCrfApproach](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfApproach) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
nerTagger = NerCrfApproach()\
    .setInputCols(["sentence", "token", "pos", "embeddings"])\
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
```

```scala
val nerTagger = new NerCrfApproach()
    .setInputCols("sentence", "token", "pos", "embeddings")
    .setLabelColumn("label")
    .setMinEpochs(1)
    .setMaxEpochs(3)
    .setC0(34)
    .setL2(3.0)
    .setOutputCol("ner")
    .fit(trainNer)
```

</div></div><div class="h3-box" markdown="1">

## NER DL (Named Entity Recognition Deep Learning annotator)

This Named Entity recognition annotator allows to train generic NER model based on Neural Networks. Its train data (train_ner) is either a labeled or an [external CoNLL 2003 IOB based](#conll-dataset) spark dataset with Annotations columns. Also the user has to provide [word embeddings annotation](#WordEmbeddings) column.  
Neural Network architecture is Char CNNs - BiLSTM - CRF that achieves state-of-the-art in most datasets.  
**Output type:** Named_Entity    
**Input types:** Document, Token, Word_Embeddings    
**Reference:** [NerDLApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ner/dl/NerDLApproach.scala) | [NerDLModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ner/dl/NerDLModel.scala)  
**Functions:**

- setLabelColumn: If DatasetPath is not provided, this Seq\[Annotation\] type of column should have labeled data per token.
- setMaxEpochs: Maximum number of epochs to train.
- setLr: Initial learning rate.
- setPo: Learning rate decay coefficient. Real Learning Rate: lr / (1 + po \* epoch).
- setBatchSize: Batch size for training.
- setDropout: Dropout coefficient.
- setVerbose: Verbosity level.
- setRandomSeed: Random seed.
- setOutputLogsPath: Folder path to save training logs.

**Note:** Please check [here](graph.md) in case you get an **IllegalArgumentException** error with a description such as:
*Graph [parameter] should be [value]: Could not find a suitable tensorflow graph for embeddings dim: [value] tags: [value] nChars: [value]. Generate graph by python code in python/tensorflow/ner/create_models before usage and use setGraphFolder Param to point to output.*

**Example:**

Refer to the [NerDLApproach](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.ner.dl.NerDLApproach) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
nerTagger = NerDLApproach()\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setLabelColumn("label")\
    .setOutputCol("ner")\
    .setMaxEpochs(10)\
    .setRandomSeed(0)\
    .setVerbose(2)
    .fit(train_ner)
```

```scala
val nerTagger = new NerDLApproach()
        .setInputCols("sentence", "token", "embeddings")
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
```

</div></div><div class="h3-box" markdown="1">

## NER Converter (Converts IOB or IOB2 representation of NER to user-friendly)

NER Converter used to finalize work of NER annotators. Combines entites with types `B-`, `I-` and etc. to the Chunks with Named entity in the metadata field (if LightPipeline is used can be extracted after `fullAnnotate()`)

This NER converter can be used to the output of a NER model into the ner chunk format.

**Output type:** Chunk
**Input types:** Document, Token, Named_Entity
**Reference:** [NerConverter](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ner/NerConverter.scala)
**Functions:**

- setWhiteList(Array(String)): If defined, list of entities to process. The rest will be ignored. Do not include IOB prefix on labels.
- setPreservePosition(Boolean): Whether to preserve the original position of the tokens in the original document or use the modified tokens.

**Example:**

Refer to the [NerConverter](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.ner.NerConverter) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
nerConverter = NerConverter()\
    .setInputCols(["sentence", "token", "ner_src"])\
    .setOutputCol("ner_chunk")
```

```scala
val nerConverter = new NerConverter()
        .setInputCols("sentence", "token", "ner_src")
        .setOutputCol("ner_chunk")
```

</div></div><div class="h3-box" markdown="1">

## Norvig SpellChecker

This annotator retrieves tokens and makes corrections automatically if not found in an English dictionary  
**Output type:** Token
**Input types:** Token
**Inputs:** Any text for corpus. A list of words for dictionary. A comma separated custom dictionary.
**Train Data:** train_corpus is a spark dataset of text content
**Reference:** [NorvigSweetingApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/norvig/NorvigSweetingApproach.scala) | [NorvigSweetingModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/norvig/NorvigSweetingModel.scala)  
**Functions:**

- setDictionary(path, tokenPattern, readAs, options): path to file with properly spelled words, tokenPattern is the regex pattern to identify them in text, readAs LINE_BY_LINE or SPARK_DATASET, with options passed to Spark reader if the latter is set.
- setCaseSensitive(boolean): defaults to false. Might affect accuracy
- setDoubleVariants(boolean): enables extra check for word combinations, more accuracy at performance
- setShortCircuit(boolean): faster but less accurate mode
- setWordSizeIgnore(int): Minimum size of word before moving on. Defaults to 3.
- setDupsLimit(int): Maximum duplicate of characters to account for. Defaults to 2.
- setReductLimit(int): Word reduction limit. Defaults to 3
- setIntersections(int): Hamming intersections to attempt. Defaults to 10.
- setVowelSwapLimit(int): Vowel swap attempts. Defaults to 6.

**Example:**

Refer to the [NorvigSweetingApproach](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingApproach) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
spell_checker = NorvigSweetingApproach() \
    .setInputCols(["token"]) \
    .setOutputCol("checked") \
    .setDictionary("coca2017.txt", "[a-zA-Z]+")
```

```scala
val symSpellChecker = new NorvigSweetingApproach()
      .setInputCols("token")
      .setOutputCol("checked")
      .setDictionary("coca2017.txt", "[a-zA-Z]+")
```

</div></div><div class="h3-box" markdown="1">

## Symmetric SpellChecker

This spell checker is inspired on Symmetric Delete algorithm. It retrieves tokens and utilizes distance metrics to compute possible derived words  
**Output type:** Token  
**Input types:** Token    
**Inputs:** Any text for corpus. A list of words for dictionary. A comma separated custom dictionary.       
**Train Data:** train_corpus is a spark dataset of text content     
**Reference:** [SymmetricDeleteApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/symmetric/SymmetricDeleteApproach.scala) | [SymmetricDeleteModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/symmetric/SymmetricDeleteModel.scala)  
**Functions:**

- setDictionary(path, tokenPattern, readAs, options): Optional dictionary of properly written words. If provided, significantly boosts spell checking performance
- setMaxEditDistance(distance): Maximum edit distance to calculate possible derived words. Defaults to 3.

**Example:**

Refer to the [SymmetricDeleteApproach](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteApproach) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
spell_checker = SymmetricDeleteApproach() \
    .setInputCols(["token"]) \
    .setOutputCol("spell") \
    .fit(train_corpus)
```

```scala
val spellChecker = new SymmetricDeleteApproach()
    .setInputCols(Array("normalized"))
    .setOutputCol("spell")
    .fit(trainCorpus)
```  

</div></div><div class="h3-box" markdown="1">

## Context SpellChecker

Implements Noisy Channel Model Spell Algorithm. Correction candidates are extracted combining context information and word information  
**Output type:** Token  
**Input types:** Token  
**Inputs:** Any text for corpus. A list of words for dictionary. A comma separated custom dictionary.      
**Train Data:** train_corpus is a spark dataset of text content    
**Reference:** [ContextSpellCheckerApproach](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/context/ContextSpellCheckerApproach.scala) | [ContextSpellCheckerModel](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/context/ContextSpellCheckerModel.scala)  
**Functions:**

- setLanguageModelClasses(languageModelClasses:Int): Number of classes to use during factorization of the softmax output in the LM. Defaults to 2000.
- setWordMaxDistance(dist:Int): Maximum distance for the generated candidates for every word. Defaults to 3.
- setMaxCandidates(candidates:Int): Maximum number of candidates for every word. Defaults to 6.
- setCaseStrategy(strategy:Int): What case combinations to try when generating candidates. ALL_UPPER_CASE = 0, FIRST_LETTER_CAPITALIZED = 1, ALL = 2. Defaults to 2.
- setErrorThreshold(threshold:Float): Threshold perplexity for a word to be considered as an error. Defaults to 10f.
- setTradeoff(alpha:Float): Tradeoff between the cost of a word error and a transition in the language model. Defaults to 18.0f.
- setMaxWindowLen(length:Integer): Maximum size for the window used to remember history prior to every correction. Defaults to 5.
- setGamma(g:Float): Controls the influence of individual word frequency in the decision.
- updateVocabClass(label:String, vocab:Array(String), append:boolean): Update existing vocabulary classes so they can cover new words. If append set to `false` overwrite vocabulary class in the model by new words, if `true` extends existing vocabulary class. Defaults to `true`.  
- updateRegexClass(label:String, regex:String): Update existing regex rule for the class defined by regex.

Train:
- setWeightedDistPath(weightedDistPath:String): The path to the file containing the weights for the levenshtein distance.
- setEpochs(epochs:Int): Number of epochs to train the language model. Defaults to 2.
- setInitialBatchSize(batchSize:Int): Batch size for the training in NLM. Defaults to 24.
- setInitialRate(initialRate:Float): Initial learning rate for the LM. Defaults to .7f.
- setFinalRate(finalRate:Float): Final learning rate for the LM. Defaults to 0.0005f.
- setValidationFraction(validationFraction:Float): Percentage of datapoints to use for validation. Defaults to .1f.
- setMinCount(minCount:Float): Min number of times a token should appear to be included in vocab. Defaults to 3.0f.
- setCompoundCount(compoundCount:Int): Min number of times a compound word should appear to be included in vocab. Defaults to 5.
- setClassCount(classCount:Int): Min number of times the word need to appear in corpus to not be considered of a special class. Defaults to 15.

**Example:**

Refer to the [ContextSpellCheckerApproach](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.spell.context.ContextSpellCheckerApproach) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
spell_checker = ContextSpellCheckerApproach() \
    .setInputCols(["token"]) \
    .setOutputCol("spell") \
    .fit(train_corpus)
```

```scala
val spellChecker = new ContextSpellCheckerApproach()
    .setInputCols(Array("token"))
    .setOutputCol("spell")
    .fit(trainCorpus)
```

</div></div><div class="h3-box" markdown="1">

## Dependency Parsers

Dependency parser provides information about word relationship. For example, dependency parsing can tell you what the subjects and objects of a verb are, as well as which words are modifying (describing) the subject. This can help you find precise answers to specific questions.
The following diagram illustrates a dependency-style analysis using the standard graphical method favored in the dependency-parsing community.

![Dependency Parser](\assets\images\dependency_parser.png)

Relations among the words are illustrated above the sentence with directed, labeled arcs from heads to dependents. We call this a typed dependency structure because the labels are drawn from a fixed inventory of grammatical relations. It also includes a root node that explicitly marks the root of the tree, the head of the entire structure. [1]

</div><div class="h3-box" markdown="1">

## Untyped Dependency Parser (Unlabeled grammatical relation)

Unlabeled parser that finds a grammatical relation between two words in a sentence. Its input is a directory with dependency treebank files.  
**Output type:** Dependency  
**Input types:** Document, POS, Token  
**Reference:** [DependencyParserApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/parser/dep/DependencyParserApproach.scala) | [DependencyParserModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/parser/dep/DependencyParserModel.scala)  
**Functions:**

- setNumberOfIterations: Number of iterations in training, converges to better accuracy
- setDependencyTreeBank: Dependency treebank folder with files in [Penn Treebank format](http://www.nltk.org/nltk_data/)
- conllU: Path to a file in [CoNLL-U format](https://universaldependencies.org/format.html)

**Example:**

Refer to the [DependencyParserApproach](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserApproach) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
dependency_parser = DependencyParserApproach() \
            .setInputCols(["sentence", "pos", "token"]) \
            .setOutputCol("dependency") \
            .setDependencyTreeBank("file://parser/dependency_treebank") \
            .setNumberOfIterations(10)
```

```scala
val dependencyParser = new DependencyParserApproach()
    .setInputCols(Array("sentence", "pos", "token"))
    .setOutputCol("dependency")
    .setDependencyTreeBank("parser/dependency_treebank")
    .setNumberOfIterations(10)
```

</div></div><div class="h3-box" markdown="1">

## Typed Dependency Parser (Labeled grammatical relation)

Labeled parser that finds a grammatical relation between two words in a sentence. Its input is a CoNLL2009 or ConllU dataset.  
**Output type:** Labeled Dependency  
**Input types:** Token, POS, Dependency  
**Reference:** [TypedDependencyParserApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/parser/typdep/TypedDependencyParserApproach.scala) | [TypedDependencyParserModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/parser/typdep/TypedDependencyParserModel.scala)  
**Functions:**

- setNumberOfIterations: Number of iterations in training, converges to better accuracy
- setConll2009: Path to a file in [CoNLL 2009 format](https://ufal.mff.cuni.cz/conll2009-st/trial-data.html)
- setConllU: Path to a file in [CoNLL-U format](https://universaldependencies.org/format.html)

**Example:**

Refer to the [TypedDependencyParserApproach](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserApproach) Scala docs for more details on the API.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
typed_dependency_parser = TypedDependencyParserApproach() \
            .setInputCols(["token", "pos", "dependency"]) \
            .setOutputCol("labdep") \
            .setConll2009("file://conll2009/eng.train") \
            .setNumberOfIterations(10)
```

```scala
val typedDependencyParser = new TypedDependencyParserApproach()
    .setInputCols(Array("token", "pos", "dependency"))
    .setOutputCol("labdep")
    .setConll2009("conll2009/eng.train"))
```

</div></div><div class="h3-box" markdown="1">

## References

[1] Speech and Language Processing. Daniel Jurafsky & James H. Martin. 2018
