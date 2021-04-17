---
layout: docs
header: true
title: Annotators
permalink: /docs/en/annotators
key: docs-annotators
modify_date: "2021-04-17"
use_language_switcher: "Python-Scala"
---

<div class="h3-box" markdown="1">

## How to read this section

All annotators in Spark NLP share a common interface, this is:

- **Annotation**: `Annotation(annotatorType, begin, end, result, meta-data,
embeddings)`
- **AnnotatorType**: some annotators share a type. This is not only
figurative, but also tells about the structure of the `metadata` map in
the Annotation. This is the one referred in the input and output of
annotators.
- **Inputs**: Represents how many and which annotator types are expected
in `setInputCols()`. These are column names of output of other annotators
in the DataFrames.
- **Output** Represents the type of the output in the column
`setOutputCol()`.

There are two types of Annotators:

- **Approach**: AnnotatorApproach extend Estimators, which are meant to be trained through `fit()`
- **Model**: AnnotatorModel extend from Transformers, which are meant to transform DataFrames through `transform()`

> **`Model`** suffix is explicitly stated when the annotator is the result of a training process. Some annotators, such as ***Tokenizer*** are transformers, but do not contain the word Model since they are not trained annotators.

`Model` annotators have a `pretrained()` on it's static object, to retrieve the public pre-trained version of a model.

- `pretrained(name, language, extra_location)` -> by default, pre-trained will bring a default model, sometimes we offer more than one model, in this case, you may have to use name, language or extra location to download them.

The types are:

|AnnotatorType|AnnotatorType|
|:---:|:---:|
|DOCUMENT = "document"|DATE = "date"|
|TOKEN = "token"|ENTITY = "entity"|
|WORDPIECE = "wordpiece"|NEGEX = "negex"|
|WORD_EMBEDDINGS = "word_embeddings"|DEPENDENCY = "dependency"|
|SENTENCE_EMBEDDINGS = "sentence_embeddings"|KEYWORD = "keyword"|
|CATEGORY = "category"|LABELED_DEPENDENCY = "labeled_dependency"|
|SENTIMENT = "sentiment"|LANGUAGE = "language"|
|POS = "pos"|CHUNK = "chunk"|
|NAMED_ENTITY = "named_entity"||

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

**Output Annotator Type:** Token  

**Input Annotator Types:** Document  

**Reference:** [Tokenizer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/Tokenizer.scala) | [TokenizerModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/TokenizerModel.scala)  

**Functions:**

- ***Parameters***
  - `caseSensitiveExceptions: BooleanParam`: Whether to care for case sensitiveness in exceptions
  - `contextChars: StringArrayParam`: Character list used to separate from token boundaries
  - `exceptions: StringArrayParam`: Words that won't be affected by tokenization rules
  - `exceptionsPath: ExternalResourceParam`: Path to file containing list of exceptions
  - `infixPatterns: StringArrayParam`: Regex patterns that match tokens within a single target. groups identify different sub-tokens. multiple defaults
  - `maxLength: IntParam`: Set the maximum allowed length for each token
  - `minLength: IntParam`: Set the minimum allowed length for each token
  - `prefixPattern: Param[String]`: Regex with groups and begins with `\\A` to match target prefix. Overrides contextCharacters Param
  - `splitChars: StringArrayParam`: Character list used to separate from the inside of tokens
  - `splitPattern: Param[String]`: Pattern to separate from the inside of tokens. takes priority over splitChars.
  - `suffixPattern: Param[String]`: Regex with groups and ends with `\\z` to match target suffix. Overrides contextCharacters Param
  - `targetPattern: Param[String]`:     Pattern to grab from text as token candidates. Defaults `\\S+`

- ***Parameter Setters***
  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setExceptions(StringArray)`: List of tokens to not alter at all. Allows composite tokens like two worded tokens that the user may not want to split.
  - `addException(String)`: Add a single exception
  - `setExceptionsPath(String)`: Path to txt file with list of token exceptions
  - `setCaseSensitiveExceptions(bool)`: Whether to follow case sensitiveness for matching exceptions in text
  - `setContextChars(StringArray)`: List of 1 character string to rip off from tokens, such as parenthesis or question marks. Ignored if using prefix, infix or suffix patterns.
  - `setSplitChars(StringArray)`: List of 1 character string to split tokens inside, such as hyphens. Ignored if using infix, prefix or suffix patterns.
  - `setSplitPattern(String)`: Regex pattern to separate from the inside of tokens. Takes priority over `setSplitChars()`.
  - `setTargetPattern(String)`: Basic regex rule to identify a candidate for tokenization. Defaults to `\\S+` which means anything not a space
  - `setSuffixPattern(String)`: Regex to identify sub-tokens that are in the end of the token. Regex has to end with `\\z` and must contain groups (). Each group will become a separate token within the prefix. Defaults to non-letter characters. e.g. quotes or parenthesis
  - `setPrefixPattern(String)`: Regex to identify sub-tokens that come in the beginning of the token. Regex has to start with `\\A` and must contain groups (). Each group will become a separate token within the prefix. Defaults to non-letter characters. e.g. quotes or parenthesis
  - `addInfixPattern(String)`: Add an extension pattern regex with groups to the top of the rules (will target first, from more specific to the more general).
  - `setMinLength(int)`: Set the minimum allowed length for each token
  - `setMaxLength(int)`: Set the maximum allowed length for each token

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getCaseSensitiveExceptions()`: Whether to follow case sensitiveness for matching exceptions in text.
  - `getContextChars()`: List of 1 character string to rip off from tokens, such as parenthesis or question marks.
  - `getInfixPatterns()`: Add an extension pattern regex with groups to the top of the rules (will target first, from more specific to the more general).
  - `getMaxLength()`: Get the maximum allowed length for each token.
  - `getMinLength()`: Get the minimum allowed length for each token.
  - `getPrefixPattern()`: Regex to identify subtokens that come in the beginning of the token. Regex has to start with `\\A` and must contain groups (). Each group will become a separate token within the prefix. Defaults to non-letter characters. e.g. quotes or parenthesis.
  - `getSplitChars()`: List of 1 character string to split tokens inside, such as hyphens. Ignored if using infix, prefix or suffix patterns.
  - `getSplitPattern()`: List of 1 character string to split tokens inside, such as hyphens. Ignored if using infix, prefix or suffix patterns.
  - `getSuffixPattern()`: Regex to identify subtokens that are in the end of the token. Regex has to end with `\\z` and must contain groups (). Each group will become a separate token within the prefix. Defaults to non-letter characters. e.g. quotes or parenthesis.
  - `getTargetPattern()`: Basic regex rule to identify a candidate for tokenization. Defaults to \\S+ which means anything not a space.

> **Note:** all these APIs receive regular expressions so please make sure that you escape special characters according to Java conventions.  

**Example:**

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
tokenizer = Tokenizer() \
    .setInputCols(["sentences"]) \
    .setOutputCol("token") \
    .setSplitChars(['-']) \
    .setContextChars(['(', ')', '?', '!']) \
    .setExceptions(["New York", "e-mail"]) \
    .setSplitPattern("'") \
    .setMaxLength(0) \
    .setMaxLength(99999) \
    .setCaseSensitiveExceptions(False)
```

```scala
val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")
    .setContextChars(Array("(", ")", "?", "!"))
    .setSplitChars(Array('-'))
    .setExceptions(["New York", "e-mail"])
    .setSplitPattern("'")
    .setMaxLength(0)
    .setMaxLength(99999)
    .setCaseSensitiveExceptions(False)
```

</div></div><div class="h3-box" markdown="1">

Refer to the [Tokenizer](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.Tokenizer) Scala docs for more details on the API.

## DocumentNormalizer (Text cleaning)

Annotator which normalizes raw text from tagged text, e.g. scraped web pages or xml documents, from document type columns into Sentence.  

**Output Annotator Type:** Document  

**Input Annotator Types:** Document  

**Reference:** [DocumentNormalizer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/DocumentNormalizer.scala)  

**Functions:**

- ***Parameters***
  - `patterns: StringArrayParam`: normalization regex patterns which match will be removed from document
  - `action: Param[String]`: action to perform applying regex patterns on text
  - `encoding: Param[String]`: file encoding to apply on normalized documents
  - `lowercase: BooleanParam`: whether to convert strings to lowercase
  - `policy: Param[String]`: removalPolicy to remove patterns from text with a given policy
  - `replacement: Param[String]`: replacement string to apply when regexes match

- ***Parameter Setters***
  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setPatterns(Array[String])`: normalization regex patterns which match will be removed from document. Defaults is `<[^>]*>`.
  - `setEncoding(String)`: Encoding to apply. Default is `UTF-8`. Valid encoding are values are: `UTF_8, UTF_16, US_ASCII, ISO-8859-1, UTF-16BE, UTF-16LE`
  - `setLowercase(Boolean)`: whether to convert strings to lowercase, default false
  - `setPolicy(String)`: removalPolicy to remove pattern from text
  - `setAction(String)`: Action to perform on text.
  - `setReplacement(String)`: Replacement string to apply when regex-es match.

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getEncoding()`: Get encoding applied to normalized documents.
  - `getLowercase()`: Lowercase tokens
  - `getPatterns()`: Regular expressions list for normalization.
  - `getPolicy()`: Get policy remove patterns from text.
  - `getReplacement()`: Replacement string to apply when regex-es match.

**Example:**

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
documentNormalizer = DocumentNormalizer() \
      .setInputCols("document") \
      .setOutputCol("normalizedDocument") \
      .setPatterns(cleanUpPatterns) \
      .setPolicy(removalPolicy)
```

```scala
    val documentNormalizer = new DocumentNormalizer()
      .setInputCols("document")
      .setOutputCol("normalizedDocument")
      .setPatterns(cleanUpPatterns)
      .setPolicy(removalPolicy)
```

</div></div><div class="h3-box" markdown="1">

Refer to the [DocumentNormalizer](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.DocumentNormalizer) Scala docs for more details on the API.

## Normalizer (Text cleaning)

Removes all dirty characters from text following a regex pattern and transforms words based on a provided dictionary  

**Output Annotator Type:** Token  

**Input Annotator Types:** Token  

**Reference:** [Normalizer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/Normalizer.scala) | [NormalizerModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/NormalizerModel.scala)

**Functions:**

- ***Parameters***
  - ` cleanupPatterns: StringArrayParam `: normalization regex patterns which match will be removed from token
  - `lowercase: BooleanParam`: whether to convert strings to lowercase
  - `slangDictionary: ExternalResourceParam`: delimited file with list of custom words to be manually corrected
  - `slangMatchCase: BooleanParam`: whether or not to be case sensitive to match slangs. Defaults to false.

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setCleanupPatterns(patterns)`: Regular expressions list for normalization, defaults \[^A-Za-z\]
  - `setLowercase(bool)`: Convert strings to lowercase tokens. Defaults to True.
  - `setSlangDictionary(value)`: txt file with delimited words to be transformed into something else
  - `setSlangMatchCase(bool)`: Whether to convert string to lowercase or not while checking

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getCleanupPatterns()`: Gets regular expressions list for normalization
  - `getLowercase()`: Returns True if input strings were converted to lowercase tokens.
  - `getSlangMatchCase()`: Whether to convert string to lowercase or not while checking.

**Example:**

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")\
    .setLowercase(True)\
    .setCleanupPatterns(["[^\w\d\s]"]) \
    .setSlangMatchCase(False)
```

```scala
val normalizer = new Normalizer()
    .setInputCols(Array("token"))
    .setOutputCol("normalized")
    .setLowercase(True)
    .setCleanupPatterns(["[^\w\d\s]"])
    .setSlangMatchCase(False)
```

</div></div><div class="h3-box" markdown="1">

Refer to the [Normalizer](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.Normalizer) Scala docs for more details on the API.

## Stemmer

Returns hard-stems out of words with the objective of retrieving the meaningful part of the word

**Output Annotator Type:** Token  

**Input Annotator Types:** Token  

**Reference:** [Stemmer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/Stemmer.scala)  

**Functions:**

- ***Parameters***
  - ` language: Param[String] `: This is the language of the text. Default is `English`

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setLanguage(string)`: This is the language of the text. Defaults to English.

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getLanguage()`: Get language for text

**Example:**

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
stemmer = Stemmer() \
    .setInputCols(["token"]) \
    .setOutputCol("stem") \
    .setLanguage("English")
    
```

```scala
val stemmer = new Stemmer()
    .setInputCols(Array("token"))
    .setOutputCol("stem")
    .setLanguage("English") 
    
```

</div></div><div class="h3-box" markdown="1">

Refer to the [Stemmer](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.Stemmer) Scala docs for more details on the API.

## Lemmatizer

Retrieves lemmas out of words with the objective of returning a base dictionary word  

**Output Annotator Type:** Token  

**Input Annotator Types:** Token

<!-- **Input**: abduct -> abducted abducting abduct abducts -->

**Reference:** [Lemmatizer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/Lemmatizer.scala) | [LemmatizerModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/LemmatizerModel.scala)  

**Functions:**

- ***Parameters***
  - `dictionary: ExternalResourceParam`: lemmatizer external dictionary, needs '`keyDelimiter`' and '`valueDelimiter`' in options for parsing target text

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setDictionary(path, keyDelimiter, valueDelimiter, readAs, options)`: Path and options to lemma dictionary, in lemma vs possible words format. readAs can be `LINE_BY_LINE` or `SPARK_DATASET`. options contain option passed to spark reader if readAs is `SPARK_DATASET`.

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getDictionary()`: Path and options to lemma dictionary, in lemma vs possible words format. readAs can be `LINE_BY_LINE` or `SPARK_DATASET`. options contain option passed to spark reader if readAs is `SPARK_DATASET`.

**Example:**

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
# Uncomment to Download the Dictionary
# !wget -q https://raw.githubusercontent.com/mahavivo/vocabulary/master/lemmas/AntBNC_lemmas_ver_001.txt

lemmatizer = Lemmatizer() \
    .setInputCols(["token"]) \
    .setOutputCol("lemma") \
    .setDictionary("./AntBNC_lemmas_ver_001.txt", value_delimiter ="\t", key_delimiter = "->")
    
```

```scala
// Uncomment to Download the Dictionary
// !wget -q https://raw.githubusercontent.com/mahavivo/vocabulary/master/lemmas/AntBNC_lemmas_ver_001.txt

val lemmatizer = new Lemmatizer()
    .setInputCols(Array("token"))
    .setOutputCol("lemma")
    .setDictionary("./AntBNC_lemmas_ver_001.txt", value_delimiter ="\t", key_delimiter = "->")
    
```

</div></div><div class="h3-box" markdown="1">

Refer to the [Lemmatizer](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.Lemmatizer) Scala docs for more details on the API.

## StopWordsCleaner

This annotator excludes from a sequence of strings (e.g. the output of a `Tokenizer()`, `Normalizer()`, `Lemmatizer()`, and `Stemmer()`) and drops all the stop words from the input sequences.

**Output Annotator Type:** token

**Input Annotator Types:** token

**Reference:** [StopWordsCleaner](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/StopWordsCleaner.scala)

**Functions:**

- ***Parameters***
  - `caseSensitive: BooleanParam`: whether to do a case-sensitive comparison over the stop words
  - `locale: Param[String]`: Locale of the input for case insensitive matching.
  - `stopWords: StringArrayParam` : the words to be filtered out. by default it's english stop words from Spark ML

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setStopWords(Array[String])`: The words to be filtered out.
  - `setCaseSensitive(Boolean)`: Whether to do a case sensitive comparison over the stop words.
  - `setLocale(String)`: Locale of the input for case insensitive matching. Ignored when `caseSensitive()` is `true`
  - `setLazyAnnotator(Boolean)`: Use `StopWordsCleaner` as a lazy annotator or not. *LazyAnnotator* is a Param in Annotators allows them to stand idle in the Pipeline and do nothing. Can be called by other Annotators in a `RecursivePipeline`

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getCaseSensitive()`: Whether to do a case sensitive comparison over the stop words.
  - `getLocale()`: Locale of the input for case insensitive matching. Ignored when `caseSensitive()` is `true`.
  - `getStopWords()`: The words to be filtered out.
  - `getLazyAnnotator()`: Whether `StopWordsCleaner` used as a lazy annotator or not. *LazyAnnotator* is a Param in Annotators allows them to stand idle in the Pipeline and do nothing. Can be called by other Annotators in a `RecursivePipeline`.

**Example:**

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
stop_words_cleaner = StopWordsCleaner() \
    .setInputCols(["token"]) \
    .setOutputCol("cleanTokens") \
    .setStopWords(["this", "is", "and"]) \
    .setCaseSensitive(False)
    
```

```scala
val stopWordsCleaner = new StopWordsCleaner()
    .setInputCols("token")
    .setOutputCol("cleanTokens")
    .setStopWords(Array("this", "is", "and"))
    .setCaseSensitive(false)
    
```

Refer to the [StopWordsCleaner](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.StopWordsCleaner) Scala docs for more details on the API.

> **NOTE:**
> If you need to `setStopWords` from a text file, you can first read and convert it into an array of string as follows.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
# your stop words text file, each line is one stop word
stopwords = sc.textFile("/tmp/stopwords/english.txt").collect()

# simply use it in StopWordsCleaner
stopWordsCleaner = StopWordsCleaner()\
      .setInputCols("token")\
      .setOutputCol("cleanTokens")\
      .setStopWords(stopwords)\
      .setCaseSensitive(False)

# or you can use pretrained models for StopWordsCleaner
stopWordsCleaner = StopWordsCleaner.pretrained()
      .setInputCols("token")\
      .setOutputCol("cleanTokens")\
      .setCaseSensitive(False)

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

// or you can use pretrained models for StopWordsCleaner
val stopWordsCleaner = StopWordsCleaner.pretrained()
      .setInputCols("token")
      .setOutputCol("cleanTokens")
      .setCaseSensitive(false)      
```

</div></div><div class="h3-box" markdown="1">

## RegexMatcher

Uses a reference file to match a set of regular expressions and put them inside a provided key. File must be comma separated.  

**Output Annotator Type:** Regex  

**Input Annotator Types:** Document  

<!-- **Input:** `the\\s\\w+`, "followed by 'the'"   -->

**Reference:** [RegexMatcher](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/RegexMatcher.scala) | [RegexMatcherModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/RegexMatcherModel.scala)  

**Functions:**

- ***Parameters***

  - `rules: ExternalResourceParam`: external resource to rules, needs 'delimiter' in options
  - `Strategy(Param[String])`: Can be any of `MATCH_FIRST | MATCH_ALL | MATCH_COMPLETE`

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setStrategy(String)`: Can be any of `MATCH_FIRST | MATCH_ALL | MATCH_COMPLETE`
  - `setRules(path, delimiter, readAs, options)`: Path to file containing a set of regex, key pair. `readAs` can be LINE_BY_LINE or SPARK_DATASET. `options` contain option passed to spark reader if `readAs` is SPARK_DATASET.
  - `setExternalRules(path, delimiter)`: Path to file containing a set of regex, key pair.

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getStrategy()`: Whether `strategy` was MATCH_FIRST | MATCH_ALL | MATCH_COMPLETE.

**Example:**

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
# For example, here are some Regex Rules which you can write in regex_rules.txt
rules = '''
renal\s\w+, started with 'renal'
cardiac\s\w+, started with 'cardiac'
\w*ly\b, ending with 'ly'
\S*\d+\S*, match any word that contains numbers
(\d+).?(\d*)\s*(mg|ml|g), match medication metrics
'''

regex_matcher = RegexMatcher() \
    .setStrategy("MATCH_ALL") \
    .setInputCols("document") \
    .setOutputCol("regex") \
    .setExternalRules(path='./regex_rules.txt', delimiter=',')
    
```

```scala
val regexMatcher = new RegexMatcher()
    .setStrategy("MATCH_ALL")
    .setInputCols(Array("document"))
    .setOutputCol("regex")
    
```

</div></div><div class="h3-box" markdown="1">

Refer to the [RegexMatcher](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.RegexMatcher) Scala docs for more details on the API.

## TextMatcher (Phrase matching)

Annotator to match entire phrases (by token) provided in a file against a Document  

**Output Annotator Type:** Entity  

**Input Annotator Types:** Document, Token

<!-- **Input**: hello world, I am looking for you -->

**Reference:** [TextMatcher](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/TextMatcher.scala) | [TextMatcherModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/TextMatcherModel.scala)  

**Functions:**

- ***Parameters***

  - `buildFromTokens: BooleanParam`: Whether the TextMatcher should take the CHUNK from TOKEN or not
  - ` caseSensitive: BooleanParam `: whether to match regardless of case.
  - `entities: ExternalResourceParam`: entities external resource.
  - ` entityValue: Param[String] `: Value for the entity metadata field
  - `mergeOverlapping: BooleanParam`: whether to merge overlapping matched chunks. Defaults false
  - `tokenizer: StructFeature[TokenizerModel]`: Tokenizer

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setBuildFromTokens(Boolean)`: Setter for buildFromTokens param
  - `setEntities(path, readAs, options)`: Provides a file with phrases to match. `readAs` gives the format of the file, can be one of {ReadAs.LINE_BY_LINE, ReadAs.SPARK_DATASET}. Defaults to LINE_BY_LINE. `option` is a map of additional parameters. Defaults to {“format”: “text”}
  - `setEntityValue(String)`: Value for the entity metadata field to indicate which chunk comes from which `textMatcher` when there are multiple `textMatchers`.
  - `setMergeOverlapping(Boolean)`: Whether to merge overlapping matched chunks. Defaults to `False`
  - `setCaseSensitive(Boolean)`: Whether to match regardless of case. Defaults to `True`
  - `setTokenizer(tokenizer)`: Tokenizer

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used.
  - `getOutputCols()`: Gets annotation column name going to generate.
  - `getBuildFromTokens()`: Getter for build From Tokens param.
  - `getEntityValue()`: whether to match regardless of case.
  - `getCaseSensitive()`: Getter for Value for the entity metadata field.
  - `getMergeOverlapping()`: Whether to merge overlapping matched chunks.

**Example:**

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
# For example, here are some entities and they are stored in sport_entities.txt
entities = ['soccer', 'world cup', 'Messi', 'FC Barcelona', 'cricket', 'Dhoni']

entity_extractor = TextMatcher() \
    .setInputCols(["inputCol"])\
    .setOutputCol("entity")\
    .setEntities("/path/to/file/sport_entities.txt") \ 
    .setEntityValue('sport_entity') \
    .setCaseSensitive(True) \ 
    .setMergeOverlapping(False)
    

```

```scala
// Assume following are our entities and they are stored in sport_entities.txt
entities = ("soccer", "world cup", "Messi", "FC Barcelona", "cricket", "Dhoni")

val entityExtractor = new TextMatcher()
    .setInputCols("inputCol")
    .setOutputCol("entity")
    .setEntities("/path/to/file/myentities.txt")
    .setEntityValue("sport_entity")
    .setCaseSensitive(true)
    .setMergeOverlapping(false)
    

```

</div></div><div class="h3-box" markdown="1">

Refer to the [TextMatcher](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.TextMatcher) Scala docs for more details on the API.

## Chunker

This annotator matches a pattern of part-of-speech tags in order to return meaningful phrases from document

**Output Annotator Type:** Chunk  

**Input Annotator Types:** Document, POS  

**Reference:** [Chunker](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/Chunker.scala)  

**Functions:**

- ***Parameters***

  - `inputAnnotatorTypes: Array[AnnotatorType]`: Input annotator type : `DOCUMENT`, `POS`
  - `outputAnnotatorType: AnnotatorType`: Output annotator type : `CHUNK`

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setRegexParsers(Array[String])`: A list of regex patterns to match chunks
  - `addRegexParser(String)`: adds a pattern to the current list of chunk patterns.

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getRegexParsers()`: A list of regex patterns to match chunks

**Example:**

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

Refer to the [Chunker](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.Chunker) Scala docs for more details on the API.

## NGramGenerator

`NGramGenerator` annotator takes as input a sequence of strings (e.g. the output of a `Tokenizer`, `Normalizer`, `Stemmer`, `Lemmatizer`, and `StopWordsCleaner`). The parameter `n` is used to determine the number of terms in each n-gram. The output will consist of a sequence of n-grams where each n-gram is represented by a space-delimited string of n consecutive words with annotatorType `CHUNK` same as the `Chunker` annotator.

**Output Annotator Type:** CHUNK  

**Input Annotator Types:** TOKEN  

**Reference:** [NGramGenerator](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/NGramGenerator.scala)  

**Functions:**

- ***Parameters***

  - `delimiter: Param[String]`: Glue character used to join the tokens
  - `enableCumulative: BooleanParam`: whether to calculate just the actual n-grams or all n-grams from 1 through n
  - `n: IntParam`: Minimum n-gram length, greater than or equal to 1.

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setN(int)`: Number elements per **n-gram (>=1)**
  - `setEnableCumulative(Boolean)`: Whether to calculate just the actual n-grams or all **n-grams from 1 through n**
  - `setDelimiter(String)`: Glue character used to join the tokens

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getDelimiter()`: Glue character used to join the tokens
  - `getEnableCumulative()`: Whether to calculate just the actual n-grams or all **n-grams from 1 through n**.
  - `getN()`: Number elements per **n-gram (>=1)**

**Example:**

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
ngrams_cum = NGramGenerator() \
            .setInputCols(["token"]) \
            .setOutputCol("ngrams") \
            .setN(2) \
            .setEnableCumulative(True) \
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

Refer to the [NGramGenerator](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.NGramGenerator) Scala docs for more details on the API.

## DateMatcher

Reads from different forms of date and time expressions and converts them to a provided date format. Extracts only ONE date per sentence. Use with sentence detector for more matches.

**Output Annotator Type:** Date  

**Input Annotator Types:** Document  

**Reference:** [DateMatcher](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/DateMatcher.scala)  

**Reads the following kind of dates:**

{:.table-model-big}
|Format|Format|Format|
|:---:|:---:|:---:|
|1978-01-28|last wednesday|5 am tomorrow|
|1984/04/02|today|0600h|
|1/02/1980|tomorrow|06:00 hours|
|2/28/79|yesterday|6pm|
|The 31st of April in the year 2008|next week at 7.30|5:30 a.m.|
|Fri, 21 Nov 1997|next week|at 5|
|Jan 21, '97|next month|12:59|
|Sun, Nov 21|next year|1988/11/23 6pm|
|jan 1st|day after|23:59|
|next thursday|the day before||
  
**Functions:**

- ***Parameters***
  - `anchorDateDay: Param[Int]`: Add an anchor year for the relative dates such as a day after tomorrow. The first day of the month has value 1 Example: 11 By default it will use the current day Default: `-1`
  - `anchorDateMonth: Param[Int]`: Add an anchor month for the relative dates such as a day after tomorrow. Month value is 1-based. e.g., 1 for January. By default it will use the current month Default: `-1`
  - `anchorDateYear: Param[Int]`: Add an anchor year for the relative dates such as a day after tomorrow.

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setDateFormat(format)`: SimpleDateFormat standard date *output* formatting. Defaults to **yyyy/MM/dd**
  - `setAnchorDateYear()`: Add an anchor year for the relative dates such as a day after tomorrow. If not set it will use the current year. Example: **2021**
  - `setAnchorDateMonth()`: Add an anchor month for the relative dates such as a day after tomorrow. If not set it will use the current month. Example: **1** which means January
  - `setAnchorDateDay()`: Add an anchor day of the day for the relative dates such as a day after tomorrow. If not set it will use the current day. Example: **11**

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getAnchorDateDay()`: Anchor day of the day for the relative dates such as a day after tomorrow. If not set it will get the current day.
  - `getAnchorDateMonth()`: Anchor month for the relative dates such as a day after tomorrow. If not set it will get the current month.
  - `getAnchorDateYear()`: Anchor year for the relative dates such as a day after tomorrow. If not set it will get the current year.

**Example:**

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

</div></div><div class="h3-box" markdown="1">

Refer to the [DateMatcher](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.DateMatcher) Scala docs for more details on the API.

## MultiDateMatcher

Reads from multiple different forms of date and time expressions and converts them to a provided date format. Extracts multiple dates per sentence.

**Output Annotator Type:** Date  

**Input Annotator Types:** Document  

**Reference:** [MultiDateMatcher](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/MultiDateMatcher.scala)  

**Reads the following kind of dates:**

{:.table-model-big}
|Format|Format|Format|
|:---:|:---:|:---:|
|1978-01-28|jan 1st|day after|
|1984/04/02|next thursday|the day before|
|1978-01-28|last wednesday|0600h|
|1988/11/23 6pm|today|06:00 hours|
|1/02/1980|tomorrow|6pm|
|2/28/79|yesterday|5:30 a.m.|
|The 31st of April in the year 2008|at 5|next week at 7.30|
|Fri, 21 Nov 1997|next week|12:59|
|Jan 21, '97|next month|23:59|
|Sun, Nov 21|next year|5 am tomorrow|
  
**Functions:**

- ***Parameters***

  - `anchorDateDay: Param[Int]`: Add an anchor year for the relative dates such as a day after tomorrow. The first day of the month has value 1 Example: 11 By default it will use the current day Default: `-1`
  - `anchorDateMonth: Param[Int]`: Add an anchor month for the relative dates such as a day after tomorrow. Month value is 1-based. e.g., 1 for January. By default it will use the current month Default: `-1`
  - `anchorDateYear: Param[Int]`: Add an anchor year for the relative dates such as a day after tomorrow. If not set it will use the current year. Example: 2021 By default it will use the current year Default: `-1`

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setDateFormat(format)`: SimpleDateFormat standard date *output* formatting. Defaults to **yyyy/MM/dd**
  - `setAnchorDateYear()`: Add an anchor year for the relative dates such as a day after tomorrow. If not set it will use the current year. Example: **2021**
  - `setAnchorDateMonth()`: Add an anchor month for the relative dates such as a day after tomorrow. If not set it will use the current month. Example: **1** which means January
  - `setAnchorDateDay()`: Add an anchor day of the day for the relative dates such as a day after tomorrow. If not set it will use the current day. Example: **11**

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getAnchorDateDay()`: Anchor day of the day for the relative dates such as a day after tomorrow. If not set it will get the current day.
  - `getAnchorDateMonth()`: Anchor month for the relative dates such as a day after tomorrow. If not set it will get the current month.
  - `getAnchorDateYear()`: Anchor year for the relative dates such as a day after tomorrow. If not set it will get the current year.

**Example:**

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

Refer to the [MultiDateMatcher](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.MultiDateMatcher) Scala docs for more details on the API.

## SentenceDetector

Finds sentence bounds in raw text. Applies rules from Pragmatic Segmenter.  

**Output Annotator Type:** Sentence

**Input Annotator Types:** Document  

**Reference:** [SentenceDetector](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/sbd/pragmatic/SentenceDetector.scala)  

**Functions:**

- ***Parameters***

  - `customBounds: StringArrayParam`: characters used to explicitly mark sentence bounds
  - `detectLists: BooleanParam`: whether take lists into consideration at sentence detection
  - `explodeSentences: BooleanParam`: whether to explode each sentence into a different row, for better parallelization. Defaults to `false`.
  - `splitLength: IntParam`: length at which sentences will be forcibly split.
  - `useAbbrevations: BooleanParam`: whether to apply abbreviations at sentence detection
  - `useCustomBoundsOnly: BooleanParam`: whether to only utilize custom bounds for sentence detection

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setCustomBounds(string)`: Custom sentence separator text
  - `setUseCustomBoundsOnly(bool)`: Use only custom bounds without considering those of Pragmatic Segmenter. Defaults to `false`. Needs customBounds.
  - `setUseAbbreviations(bool)`: Whether to consider abbreviation strategies for better accuracy but slower performance. Defaults to `true`.
  - `setExplodeSentences(bool)`: Whether to split sentences into different Dataset rows. Useful for higher parallelism in fat rows. Defaults to `false`.

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getCustomBounds()`: Custom sentence separator text
  - `getExplodeSentences()`: Whether to split sentences into different Dataset rows.
  - `getUseAbbreviations()`: Whether to consider abbreviation strategies for better accuracy but slower performance.
  - `getsetUseCustomBoundsOnly()`: Use only custom bounds without considering those of Pragmatic Segmenter.

**Example:**

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

Refer to the [SentenceDetector](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector) Scala docs for more details on the API.

## POSTagger (Part of speech tagger)

Sets a POS tag to each word within a sentence. Its train data (train_pos) is a spark dataset of [POS format values](#TrainPOS) with Annotation columns.  

**Output Annotator Type:** POS  

**Input Annotator Types:** Document, Token  

**Reference:** [PerceptronApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/pos/perceptron/PerceptronApproach.scala) | [PerceptronModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/pos/perceptron/PerceptronModel.scala)  

**Functions:**

- ***Parameters***

  - `ambiguityThreshold: DoubleParam`: How much percentage of total amount of words are covered to be marked as frequent
  - `frequencyThreshold: IntParam`: How many times at least a tag on a word to be marked as frequent
  - `nIterations: IntParam`: Number of iterations in training, converges to better accuracy
  - `posCol: Param[String]`: Column of Array of POS tags that match tokens

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setAmbiguityThreshold(Double)`: Setter for how much percentage of total amount of words are covered to be marked as frequent
  - `setFrequencyThreshold(int)`: Setter for how many times at least a tag on a word to be marked as frequent
  - `setNIterations(number)`: Number of iterations for training. May improve accuracy but takes longer. Default **5**
  - `setPosColumn(colname)`: Column containing an array of POS Tags matching every token on the line.

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getNIterations()`: Number of iterations for training. May improve accuracy but takes longer. Default **5**.

**Example:**

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
pos_tagger = PerceptronApproach() \
    .setInputCols(["token", "sentence"]) \
    .setOutputCol("pos") \
    .setNIterations(2) \
    .setFrequencyThreshold(30)
    
```

```scala
val posTagger = new PerceptronApproach()
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("pos")
    .setNIterations(2)
    .setFrequencyThreshold(30)
    
```

</div></div><div class="h3-box" markdown="1">

Refer to the [PerceptronApproach](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronApproach) Scala docs for more details on the API.

## ViveknSentimentDetector

Scores a sentence for a sentiment
  
**Output Annotator Type:** sentiment  

**Input Annotator Types:** Document, Token  

<!-- **Input:** File or folder of text files of positive and negative data   -->

**Reference:** [ViveknSentimentApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/sda/vivekn/ViveknSentimentApproach.scala) | [ViveknSentimentModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/sda/vivekn/ViveknSentimentModel.scala)
  
**Functions:**

- ***Parameters***

  - `pruneCorpus: IntParam`: Removes unfrequent scenarios from scope. The higher the better performance. Defaults `1`
  - `sentimentCol: Param[String]`: column with the sentiment result of every row. Must be '`positive`' or '`negative`'
  - `importantFeatureRatio(Double)`: Proportion to lookahead in unimportant features.
  - `unimportantFeatureStep(Double)`: Proportion to lookahead in unimportant features.

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setImportantFeatureRatio(Double)`: Set Proportion of feature content to be considered relevant. Defaults to **0.5**
  - `setUnimportantFeatureStep()`: Set Proportion to lookahead in unimportant features. Defaults to **0.025**
  - `setSentimentCol(String)`: Column with the sentiment result of every row. Must be 'positive' or 'negative'
  - `setCorpusPrune(true)`: When training on small data you may want to disable this to not cut off infrequent words
  - `setFeatureLimit()`: Set content feature limit, to boost performance in very dirt text. Default disabled with **-1**.

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getFeatureLimit()`: Get content feature limit, to boost performance in very dirt text. Default disabled with **-1**
  - `getImportantFeatureRatio()`: Get Proportion of feature content to be considered relevant. Defaults to **0.5**
  - `getUnimportantFeatureStep()`: Get Proportion to lookahead in unimportant features. Defaults to **0.025**

**Example:**

- **Train your own model**

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
sentiment_detector = ViveknSentimentApproach() \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("sentiment") \
    .setSentimentCol("sentiment_label") \
    .setCorpusPrune(0) \
    .setImportantFeatureRatio(16.66)
    
```

```scala
val sentimentDetector = new ViveknSentimentApproach()
    .setInputCols(Array("token", "sentence"))
    .setOutputCol("vivekn")
    .setSentimentCol("sentiment_label")
    .setCorpusPrune(0)
    .setImportantFeatureRatio(16.66)
    
```

</div>

- **Use a pretrained model**

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
sentiment_detector = ViveknSentimentModel.pretrained() \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("sentiment") \
    
```

```scala
val sentimentDetector = new ViveknSentimentModel.pretrained
    .setInputCols(Array("token", "sentence"))
    .setOutputCol("vivekn")
    
```

</div></div><div class="h3-box" markdown="1">

Refer to the [ViveknSentimentApproach](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentApproach) Scala docs for more details on the API.

## SentimentDetector (Sentiment analysis)

Scores a sentence for a sentiment  

**Output Annotator Type:** Sentiment  

**Input Annotator Types:** Document, Token  

**Reference:** [SentimentDetector](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/sda/pragmatic/SentimentDetector.scala) | [SentimentDetectorModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/sda/pragmatic/SentimentDetectorModel.scala)  

**Functions:**

- ***Parameters***

  - `decrementMultiplier: DoubleParam`: multiplier for decrement sentiments. Defaults **-2.0**
  - `dictionary: ExternalResourceParam`: delimited file with a list sentiment tags per word. Requires `delimiter` in options
  - `enableScore: BooleanParam`: if true, score will show as the double value, else will output string `positive` or `negative`. Defaults `false`
  - `incrementMultiplier: DoubleParam`: multiplier for increment sentiments.
  - `negativeMultiplier: DoubleParam`: "multiplier for negative sentiments. Defaults **-1.0**
  - `positiveMultiplier: DoubleParam`: multiplier for positive sentiments. Defaults **1.0**
  - `reverseMultiplier: DoubleParam`: multiplier for revert sentiments. Defaults **-1.0**

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setDictionary(path, delimiter, readAs, options)`: *path* to file with list of inputs and their content, with such delimiter, *readAs* `LINE_BY_LINE` or as `SPARK_DATASET`. If latter is set, *options* is passed to spark reader.
  - `setPositiveMultiplier(double)`: Defaults to **1.0**
  - `setNegativeMultiplier(double)`: Defaults to **-1.0**
  - `setIncrementMultiplier(double)`: Defaults to **2.0**
  - `setDecrementMultiplier(double)`: Defaults to **-2.0**
  - `setReverseMultiplier(double)`: Defaults to **-1.0**

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate

<!-- **Input:**
- superb,positive
- bad,negative
- lack of,revert
- very,increment
- barely,decrement -->

**Example:**

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
sentiment_detector = SentimentDetector() \
    .setInputCols(["token", "sentence"]) \
    .setOutputCol("sentiment") \
    .setPositiveMultiplier(1.0)\
    .setNegativeMultiplier(-1.0)\
    .setIncrementMultiplier(2.0)\
    .setDecrementMultiplier(-2.0)\
    .setReverseMultiplier(-1.0)
```

```scala
val sentimentDetector = new SentimentDetector
    .setInputCols(Array("token", "sentence"))
    .setOutputCol("sentiment")
    .setPositiveMultiplier(1.0)
    .setNegativeMultiplier(-1.0)
    .setIncrementMultiplier(2.0)
    .setDecrementMultiplier(-2.0)
    .setReverseMultiplier(-1.0)
```

</div></div><div class="h3-box" markdown="1">

Refer to the [SentimentDetector](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetector) Scala docs for more details on the API.

## WordEmbeddings

Word Embeddings lookup annotator that maps tokens to vectors  

**Output Annotator Type:** Word_Embeddings  

**Input Annotator Types:** Document, Token  

**Reference:**  [WordEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/WordEmbeddings.scala) | [WordEmbeddingsModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/WordEmbeddingsModel.scala)  

**Functions:**

- ***Parameters***

  - `readCacheSize: IntParam`: cache size for items retrieved from storage. Increase for performance but higher memory consumption
  - `writeBufferSize: IntParam`: buffer size limit before dumping to disk storage while writing

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setReadCacheSize(int)`: Cache size for items retrieved from storage. Increase for performance but higher memory consumption.
  - `setWriteBufferSize(int)`: Buffer size limit before dumping to disk storage while writing.
  - `setStoragePath(path, format)`: sets [word embeddings](https://en.wikipedia.org/wiki/Word_embedding) options.
    - *path*: word embeddings file  
    - *format*: format of word embeddings files:
      - *TEXT* -> This format is usually used by [Glove](https://nlp.stanford.edu/projects/glove/)
      - *BINARY* -> This format is usually used by [Word2Vec](https://code.google.com/archive/p/word2vec/)
  - `setCaseSensitive`: whether to ignore case in tokens for embeddings matching

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate

**Example:**

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
embeddings = WordEmbeddings()
    .setStoragePath("/tmp/glove.6B.100d.txt", "TEXT")\
    .setDimension(100)\
    .setStorageRef("glove_100d") \
    .setInputCols("document", "token") \
    .setOutputCol("embeddings")


# or you can use the pretrained models for WordEmbeddings
embeddings = WordEmbeddingsModel.pretrained()
    .setInputCols("document", "token") \
    .setOutputCol("embeddings")

```

```scala
val embeddings = new WordEmbeddings()
    .setStoragePath("/tmp/glove.6B.100d.txt", "TEXT")
    .setDimension(100)
    .setStorageRef("glove_100d")
    .setInputCols("document", "token")
    .setOutputCol("embeddings")

// or you can use the pretrained models for WordEmbeddings
val embeddings = WordEmbeddingsModel.pretrained()
    .setInputCols("document", "token")
    .setOutputCol("embeddings")
```

</div>

There are also two convenient functions to retrieve the embeddings coverage with respect to the transformed dataset:  

- `withCoverageColumn(dataset, embeddingsCol, outputCol)`: Adds a custom column with **word coverage** stats for the embedded field: (coveredWords, totalWords, coveragePercentage). This creates a new column with statistics for each row.
- `overallCoverage(dataset, embeddingsCol)`: Calculates overall **word coverage** for the whole data in the embedded field. This returns a single coverage object considering all rows in the field.

</div><div class="h3-box" markdown="1">

Refer to the [WordEmbeddings](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.embeddings.WordEmbeddings) Scala docs for more details on the API.

## BertEmbeddings

BERT (Bidirectional Encoder Representations from Transformers) provides dense vector representations for natural language by using a deep, pre-trained neural network with the Transformer architecture

You can find the pre-trained models for `BertEmbeddings` in the [Spark NLP Models](https://github.com/JohnSnowLabs/spark-nlp-models) repository

**Output Annotator Type:** Word_Embeddings  

**Input Annotator Types:** Document, Token

**Reference:** [BertEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/BertEmbeddings.scala)  

**Functions:**

- ***Parameters***

  - `batchSize: IntParam`: Batch size. Large values allows faster processing but requires more memory.
  - `configProtoBytes: IntArrayParam`: ConfigProto from tensorflow, serialized into byte array. Get with `config_proto.SerializeToString()`
  - `maxSentenceLength: IntParam`: Max sentence length to process
  - `vocabulary: MapFeature[String, Int]`: vocabulary

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setCaseSensitive(Boolean)`: Whether to lowercase tokens or not
  - `setConfigProtoBytes(Array[int])`: ConfigProto from tensorflow, serialized into byte array. Get with `config_proto.SerializeToString()`
  - `setDimension(int)`: Set Embeddings dimensions for the BERT model Only possible to set this when the first time is saved dimension is not changeable, it comes from BERT config file
  - `setMaxSentenceLength(int)`: Max sentence length to process
  - `setVocabulary(Map[String, Int])`: Vocabulary used to encode the words to ids with `WordPieceEncoder`

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getConfigProtoBytes()`: ConfigProto from tensorflow, serialized into byte array. Get with `config_proto.SerializeToString()`
  - `getMaxSentenceLength()`: Max sentence length to process
  - `getCaseSensitive()`: Whether to follow case sensitiveness for matching exceptions in text.
  - `getDimension()`: Getter for Embeddings dimensions for the BERT model.

**Example:**

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

Refer to the [BertEmbeddings](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.embeddings.BertEmbeddings) Scala docs for more

## BertSentenceEmbeddings

BERT (Bidirectional Encoder Representations from Transformers) provides dense vector representations for natural language by using a deep, pre-trained neural network with the Transformer architecture

You can find the pre-trained models for `BertEmbeddings` in the [Spark NLP Models](https://github.com/JohnSnowLabs/spark-nlp-models) repository

**Output Annotator Type:** Sentence_Embeddings  

**Input Annotator Types:** Document

**Reference:** [BertSentenceEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/BertSentenceEmbeddings.scala)  

**Functions:**

- ***Parameters***

  - `batchSize: IntParam`: Batch size. Large values allows faster processing but requires more memory.
  - `configProtoBytes: IntArrayParam`: ConfigProto from tensorflow, serialized into byte array. Get with `config_proto.SerializeToString()`
  - `isLong: BooleanParam`: Use Long type instead of Int type for inputs
  - `maxSentenceLength: IntParam` : Max sentence length to process
  - `vocabulary: MapFeature[String, Int]`: vocabulary

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setCaseSensitive(Boolean)`: Whether to lowercase tokens or not
  - `setConfigProtoBytes(Array[Int])`: ConfigProto from tensorflow, serialized into byte array.
  - `setDimension(int)`: Set Embeddings dimensions for the BERT model Only possible to set this when the first time is saved dimension is not changeable, it comes from BERT config file
  - `setMaxSentenceLength(int)`: Max sentence length to process
  - `setVocabulary(Map[String, Int])`: Vocabulary used to encode the words to ids with `WordPieceEncoder`

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getConfigProtoBytes()`: ConfigProto from tensorflow, serialized into byte array. Get with `config_proto.SerializeToString()`
  - `getMaxSentenceLength()`: Max sentence length to process
  - `getCaseSensitive()`: Whether to follow case sensitiveness for matching exceptions in text.
  - `getDimension()`: Getter for Embeddings dimensions for the BERT model.

**Example:**

How to use pretrained BertEmbeddings:

<div class="tabs-box" markdown="1">

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

Refer to the [BertSentenceEmbeddings](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.embeddings.BertSentenceEmbeddings) Scala docs for more

## ElmoEmbeddings

Computes contextualized word representations using character-based word representations and bidirectional LSTMs

You can find the pre-trained model for `ElmoEmbeddings` in the  [Spark NLP Models](https://github.com/JohnSnowLabs/spark-nlp-models#english---models) repository

**Output Annotator Type:** Word_Embeddings

**Input Annotator Types:** Document, Token

**Reference:** [ElmoEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/ElmoEmbeddings.scala)  

**Functions:**

- ***Parameters***

  - `batchSize: IntParam`: Batch size. Large values allows faster processing but requires more memory.
  - `configProtoBytes: IntArrayParam`: ConfigProto from tensorflow, serialized into byte array.
  - `poolingLayer: Param[String]`: Set ELMO pooling layer to: `word_emb`, `lstm_outputs1`, `lstm_outputs2`, or `elmo`

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setBatchSize(int)`: Large values allows faster processing but requires more memory.
  - `setConfigProtoBytes(Array[Int])`: ConfigProto from tensorflow, serialized into byte array. Get with `config_proto.SerializeToString()`
  - `setCaseSensitive(Boolean)`: Whether to lowercase tokens or not
  - `setDimension(int)`: Set Dimension of pooling layer. This is meta for the annotation and will not affect the actual embedding calculation.
  - `setPoolingLayer(String)`: Function used to set the embedding output layer of the ELMO model word_emb: the character-based word representations with shape `[batch_size, max_length, 512]`.

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getPoolingLayer()`: Function used to get the embedding output layer of the ELMO model word_emb: the character-based word representations with shape `[batch_size, max_length, 512]`.
  - `getConfigProtoBytes()`: ConfigProto from tensorflow, serialized into byte array. Get with `config_proto.SerializeToString()`
  - `getCaseSensitive()`: Whether to follow case sensitiveness for matching exceptions in text.
  - `getDimension()`: Getter for Embeddings dimensions for the BERT model.

**Example:**

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

Refer to the [ElmoEmbeddings](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.embeddings.ElmoEmbeddings) Scala docs for more

## AlbertEmbeddings

Computes contextualized word representations using "A Lite" implementation of BERT algorithm by applying parameter-reduction techniques

You can find the pre-trained model for `AlbertEmbeddings` in the  [Spark NLP Models](https://github.com/JohnSnowLabs/spark-nlp-models#english---models) repository

**Output Annotator Type:** Word_Embeddings

**Input Annotator Types:** Document, Token

**Reference:** [AlbertEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/AlbertEmbeddings.scala)  

**Functions:**

- ***Parameters***

  - `batchSize: IntParam`: Batch size. Large values allows faster processing but requires more memory
  - `caseSensitive: BooleanParam` : whether to ignore case in tokens for embeddings matching
  - `dimension: IntParam`: Number of embedding dimensions
  - `configProtoBytes: IntArrayParam`: ConfigProto from tensorflow, serialized into byte array.
  - `maxSentenceLength: IntParam`: Max sentence length to process

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setBatchSize(int)`: Batch size. Large values allows faster processing but requires more memory.
  - `setMaxSentenceLength(int)`: Max sentence length to process
  - `setDimension(int)`: Number of embedding dimensions
  - `setCaseSensitive(Boolean)`: whether to ignore case in tokens for embeddings matching

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getCaseSensitive()`: whether to ignore case in tokens for embeddings matching
  - `getDimension()`: Number of embedding dimensions

**Examples:**

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

Refer to the [AlbertEmbeddings](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.embeddings.AlbertEmbeddings) Scala docs for more

## XlnetEmbeddings

Computes contextualized word representations using combination of Autoregressive Language Model and Permutation Language Model

You can find the pre-trained model for `XlnetEmbeddings` in the  [Spark NLP Models](https://github.com/JohnSnowLabs/spark-nlp-models#english---models) repository

**Output Annotator Type:** Word_Embeddings

**Input Annotator Types:** Document, Token

**Reference:** [XlnetEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/XlnetEmbeddings.scala)  

**Functions:**

- ***Parameters***

  - `batchSize: IntParam`: Batch size. Large values allows faster processing but requires more memory
  - `caseSensitive: BooleanParam` : whether to ignore case in tokens for embeddings matching
  - `modelIfNotSet`: XLNet tensorflow Model.
  - `dimension: IntParam`: Number of embedding dimensions
  - `configProtoBytes: IntArrayParam`: ConfigProto from tensorflow, serialized into byte array.
  - `maxSentenceLength: IntParam`: Max sentence length to process

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setBatchSize(int)`: Batch size. Large values allows faster processing but requires more memory.
  - `setMaxSentenceLength(int)`: Max sentence length to process.
  - `setModelIfNotSet()`: Sets `XLNet` tensorflow Model.
  - `setDimension(value: Int)`: Set dimension of Embeddings Since output shape depends on the model selected, see `https://github.com/zihangdai/xlnet`for further reference

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getMaxSentenceLength()`: Max sentence length to process
  - `getModelIfNotSet()`: Gets XLNet tensorflow Model

**Example:**

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

Refer to the [XlnetEmbeddings](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.embeddings.XlnetEmbeddings) Scala docs for more

## UniversalSentenceEncoder

The Universal Sentence Encoder encodes text into high dimensional vectors that can be used for text classification, semantic similarity, clustering and other natural language tasks.

**Output Annotator Type:** SENTENCE_EMBEDDINGS

**Input Annotator Types:** Document

**Reference:** [UniversalSentenceEncoder](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/UniversalSentenceEncoder.scala)

**Functions:**

- ***Parameters***

  - `modelIfNotSet`: XLNet tensorflow Model.
  - `dimension: IntParam`: Number of embedding dimensions
  - `configProtoBytes: IntArrayParam`: ConfigProto from tensorflow, serialized into byte array.

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setConfigProtoBytes(bytes: Array[Int])`: ConfigProto from tensorflow, serialized into byte array. Get with `config_proto.SerializeToString()`
  - `setLoadSP(value: Boolean)`: set loadSP

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getConfigProtoBytes()`: ConfigProto from tensorflow, serialized into byte array.
  - `getLoadSP()`: Getter for LoadSP

**Example:**

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
use = UniversalSentenceEncoder.pretrained() \
        .setInputCols("sentence") \
        .setOutputCol("use_embeddings")
```

```scala
val use = UniversalSentenceEncoder.pretrained()
        .setInputCols("document")
        .setOutputCol("use_embeddings")
```

</div></div><div class="h3-box" markdown="1">

Refer to the [UniversalSentenceEncoder](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder) Scala docs for more

## SentenceEmbeddings

This annotator converts the results from `WordEmbeddings`, `BertEmbeddings`, `ElmoEmbeddings`, `AlbertEmbeddings`, or `XlnetEmbeddings` into `sentence` or `document` embeddings by either summing up or averaging all the word embeddings in a sentence or a document (depending on the `inputCols`).

**Output Annotator Type:** SENTENCE_EMBEDDINGS

**Input Annotator Types:** Document

**Reference:** [SentenceEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/SentenceEmbeddings.scala)

**Functions:**

- ***Parameters***

  - `dimension: IntParam`: Number of embedding dimensions
  - `poolingStrategy: Param[String]`: Choose how you would like to aggregate Word Embeddings to Sentence Embeddings: AVERAGE or SUM

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setPoolingStrategy()`: Choose how you would like to aggregate Word Embeddings to Sentence Embeddings: `AVERAGE` or `SUM`
  - `setDimension(int)`: Number of embedding dimensions

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getDimension()`: Number of embedding dimensions

**Example:**

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

> **NOTE:** If you choose `document` as your input for `Tokenizer`, `WordEmbeddings/BertEmbeddings`, and `SentenceEmbeddings` then it averages/sums all the embeddings into one array of embeddings. However, if you choose `sentence` as `inputCols` then for each sentence `SentenceEmbeddings` generates one array of embeddings.

> **TIP:** Here is how you can explode and convert these embeddings into `Vectors` or what's known as `Feature` column so it can be used in Spark ML regression or clustering functions

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

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

Refer to the [SentenceEmbeddings](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.embeddings.SentenceEmbeddings) Scala docs for more

## ChunkEmbeddings

This annotator utilizes `WordEmbeddings` or `BertEmbeddings` to generate chunk embeddings from either `Chunker`, `NGramGenerator`, or `NerConverter` outputs.

**Output Annotator Type:** CHUNK

**Input Annotator Types:** CHUNK, Word_Embeddings

**Reference:** [ChunkEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/ChunkEmbeddings.scala)

**Functions:**

- ***Parameters***

  - `skipOOV: BooleanParam`: Whether to discard default vectors for OOV words from the aggregation / pooling
  - `poolingStrategy: Param[String]`: Choose how you would like to aggregate Word Embeddings to Chunk Embeddings: AVERAGE or SUM

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setPoolingStrategy(String)`: Choose how you would like to aggregate Word Embeddings to Sentence Embeddings: AVERAGE or SUM
  - `setSkipOOV(Boolean)`: Whether to discard default vectors for OOV words from the aggregation / pooling

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getPoolingStrategy()`: Choose how you would like to aggregate Word Embeddings to Chunk Embeddings: AVERAGE or SUM
  - `getSkipOOV()`: Whether to discard default vectors for OOV words from the aggregation / pooling

**Example:**

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

> **TIP:** Here is how you can explode and convert these embeddings into `Vectors` or what's known as `Feature` column so it can be used in Spark ML regression or clustering functions

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

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

Refer to the [ChunkEmbeddings](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.embeddings.ChunkEmbeddings) Scala docs for more

## ClassifierDL (Multi-class Text Classification)

ClassifierDL is a generic Multi-class Text Classification. ClassifierDL uses the state-of-the-art Universal Sentence Encoder as an input for text classifications. The ClassifierDL annotator uses a deep learning model (DNNs) we have built inside TensorFlow and supports up to 100 classes

**Output Annotator Type:** CATEGORY

**Input Annotator Types:** SENTENCE_EMBEDDINGS

**Reference:** [ClassifierDLApproach](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/ClassifierDLApproach.scala) | [ClassifierDLModel](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/ClassifierDLModel.scala)

**Functions:**

- ***Parameters***

  - `batchSize: IntParam`: Batch size
  - `configProtoBytes: IntArrayParam` : ConfigProto from tensorflow, serialized into byte array. Get with `config_proto.SerializeToString()`
  - `dropout: FloatParam` : Dropout coefficient
  - `enableOutputLogs: BooleanParam` : Whether to output to annotators log folder
  - `labelColumn: Param[String]`: Column with label per each document
  - `lr: FloatParam`: Learning Rate
  - `maxEpochs: IntParam`: Maximum number of epochs to train
  - `randomSeed: IntParam`: Random seed
  - `validationSplit: FloatParam` : Choose the proportion of training dataset to be validated against the model on each Epoch.
  - `verbose: IntParam`: Level of verbosity during training

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setLabelColumn`: If DatasetPath is not provided, this Seq\[Annotation\] type of column should have labeled data per token.
  - `setLr(float)`: Initial learning rate.
  - `setBatchSize(int)`: Batch size for training.
  - `setDropout(float)`: Dropout coefficient.
  - `setMaxEpochs(int)`: Maximum number of epochs to train.
  - `setEnableOutputLogs(Boolean)`: Whether to output to annotators log folder.
  - `setValidationSplit(float)`: Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between **0.0** and **1.0** and by default it is **0.0** and off.
  - `setVerbose(int)`: Level of verbosity during training.
  - `setOutputLogsPath(String)`: Folder path to save training logs.

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getClasses()`: get the tags used to trained this `NerDLModel`
  - `getConfigProtoBytes()`: Tensorflow config Protobytes passed to the TF session

> **NOTE**: This annotator accepts a label column of a single item in either type of String, Int, Float, or Double.

> **NOTE**: UniversalSentenceEncoder, BertSentenceEmbeddings, or SentenceEmbeddings can be used for the inputCol

**Example:**

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

Refer to the [ClassifierDLApproach](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLApproach) | [ClassifierDLModel](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLModel) Scala docs for more

## MultiClassifierDL (Multi-label Text Classification)

 MultiClassifierDL is a Multi-label Text Classification. MultiClassifierDL uses a Bidirectional GRU with Convolution model that we have built inside TensorFlow and supports up to 100 classes. The input to MultiClassifierDL is Sentence Embeddings such as state-of-the-art UniversalSentenceEncoder, BertSentenceEmbeddings, or SentenceEmbeddings

**Output Annotator Type:** CATEGORY

**Input Annotator Types:** SENTENCE_EMBEDDINGS

**Reference:** [MultiClassifierDLApproach](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/MultiClassifierDLApproach.scala) | [MultiClassifierDLModel](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/MultiClassifierDLModel.scala)

**Functions:**

- ***Parameters***

  - `batchSize: IntParam`: Batch size
  - `configProtoBytes: IntArrayParam` : ConfigProto from tensorflow, serialized into byte array. Get with `config_proto.SerializeToString()`
  - `enableOutputLogs: BooleanParam` : Whether to output to annotators log folder
  - `labelColumn: Param[String]`: Column with label per each document
  - `lr: FloatParam`: Learning Rate
  - `maxEpochs: IntParam`: Maximum number of epochs to train
  - `randomSeed: IntParam`: Random seed
  - `shufflePerEpoch: BooleanParam`: Whether to shuffle the training data on each Epoch
  - `threshold: FloatParam`: The minimum threshold for each label to be accepted. Default is **0.5**
  - `validationSplit: FloatParam` : Choose the proportion of training dataset to be validated against the model on each Epoch.
  - `verbose: IntParam`: Level of verbosity during training

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setLabelColumn(String)`: If DatasetPath is not provided, this Seq\[Annotation\] type of column should have labeled data per token.
  - `setLr(float)`: Initial learning rate.
  - `setBatchSize(int)`: Batch size for training.
  - `setMaxEpochs(int)`: Maximum number of epochs to train.
  - `setEnableOutputLogs(Boolean)`: Whether to output to annotators log folder.
  - `setValidationSplit(float)`: Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between **0.0** and **1.0** and by default it is **0.0** and off.
  - `setVerbose(int)`: Level of verbosity during training.
  - `setOutputLogsPath(String)`: Folder path to save training logs.

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getBatchSize()`: Getter for Batch size
  - `getConfigProtoBytes()`: Tensorflow config Protobytes passed to the TF session
  - `getEnableOutputLogs()`: Whether to output to annotators log folder
  - `getLabelColumn()`: Column with label per each document
  - `getLr()`: Getter for Learning Rate
  - `getMaxEpochs()`: Maximum number of epochs to train
  - `getShufflePerEpoch()`: Max sequence length to feed into TensorFlow
  - `getThreshold()`: The minimum threshold for each label to be accepted.
  - `getValidationSplit()`: Choose the proportion of training dataset to be validated against the model on each Epoch.

> **NOTE**: This annotator accepts a label column of a single item in either type of String, Int, Float, or Double.

> **NOTE**: UniversalSentenceEncoder, BertSentenceEmbeddings, or SentenceEmbeddings can be used for the inputCol

**Example:**

<div class="tabs-box" markdown="1">

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

Refer to the [MultiClassifierDLApproach](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.classifier.dl.MultiClassifierDLApproach) | [MultiClassifierDLModel](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.classifier.dl.MultiClassifierDLModel) Scala docs for more

## SentimentDL (Multi-class Sentiment Analysis annotator)

SentimentDL is an annotator for multi-class sentiment analysis. This annotator comes with 2 available pre-trained models trained on IMDB and Twitter datasets

**Output Annotator Type:** CATEGORY

**Input Annotator Types:** SENTENCE_EMBEDDINGS

**Reference:** [SentimentDLApproach](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/SentimentDLApproach.scala) | [SentimentDLModel](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/SentimentDLModel.scala)

**Functions:**

- ***Parameters***

  - `batchSize: IntParam`: Batch size
  - `configProtoBytes: IntArrayParam` : ConfigProto from tensorflow, serialized into byte array. Get with `config_proto.SerializeToString()`
  - `dropout: FloatParam` : Dropout coefficient
  - `enableOutputLogs: BooleanParam` : Whether to output to annotators log folder
  - `labelColumn: Param[String]`: Column with label per each document
  - `lr: FloatParam`: Learning Rate
  - `maxEpochs: IntParam`: Maximum number of epochs to train
  - `randomSeed: IntParam`: Random seed
  - `shufflePerEpoch: BooleanParam`: Whether to shuffle the training data on each Epoch
  - `threshold: FloatParam`: The minimum threshold for each label to be accepted. Default is **0.5**
  - `validationSplit: FloatParam` : Choose the proportion of training dataset to be validated against the model on each Epoch.
  - `verbose: IntParam`: Level of verbosity during training

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setLabelColumn()`: If DatasetPath is not provided, this Seq\[Annotation\] type of column should have labeled data per token.
  - `setLr(Float)`: Initial learning rate.
  - `setBatchSize(int)`: Batch size for training.
  - `setDropout(Float)`: Dropout coefficient.
  - `setThreshold(Float)`: The minimum threshold for the final result otherwise it will be either neutral or the value set in thresholdLabel.
  - `setThresholdLabel(String)`: In case the score is less than threshold, what should be the label. Default is neutral.
  - `setMaxEpochs(int)`: Maximum number of epochs to train.
  - `setEnableOutputLogs(Boolean)`: Whether to output to annotators log folder.
  - `setOutputLogsPath(String)`: Folder path to save training logs.
  - `setValidationSplit(Float)`: Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between **0.0** and **1.0** and by default it is **0.0** and off.
  - `setVerbose(int)`: Level of verbosity during training.

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getLazyAnnotator()`: Whether `SentimentDL` used as LazyAnnotator or not. *LazyAnnotator* is a Param in Annotators allows them to stand idle in the Pipeline and do nothing. Can be called by other Annotators in a `RecursivePipeline`.

> **NOTE**: This annotator accepts a label column of a single item in either type of String, Int, Float, or Double.

> **NOTE**: UniversalSentenceEncoder, BertSentenceEmbeddings, or SentenceEmbeddings can be used for the inputCol

**Example:**

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

Refer to the [SentimentDLApproach](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.classifier.dl.SentimentDLApproach) | [SentimentDLModel](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.classifier.dl.SentimentDLModel) Scala docs for more

## LanguageDetectorDL (Language Detection and Identiffication)

LanguageDetectorDL is a state-of-the-art language detection and identification annotator trained by using TensorFlow/keras neural networks.

**Output Annotator Type:** LANGUAGE

**Input Annotator Types:** DOCUMENT or SENTENCE

**Reference:** [LanguageDetectorDL](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ld/dl/LanguageDetectorDL.scala)

**Functions:**

- ***Parameters***

  - `coalesceSentences: BooleanParam` : coalesceSentences, output of all sentences will be averaged to one output instead of one output per sentence
  - `threshold: FloatParam` : The minimum threshold for each label to be accepted.
  - `language: MapFeature[String, Int]`: language
  - `configProtoBytes: IntArrayParam`: ConfigProto from tensorflow, serialized into byte array.

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setThreshold`: The minimum threshold for the final result otheriwse it will be either neutral or the value set in thresholdLabel.
  - `setThresholdLabel`: In case the score is less than threshold, what should be the label. Default is Unknown.
  - `setCoalesceSentences`: If sets to true the output of all sentences will be averaged to one output instead of one output per sentence. Default to true.

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getConfigProtoBytes()`: ConfigProto from tensorflow, serialized into byte array. Get with `config_proto.SerializeToString()`
  - `getLanguage()`: Getter for languages
  - `getThreshold()`: Getter for Threshold

**Example:**

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

Refer to the [LanguageDetectorDL](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.ld.dl.LanguageDetectorDL) Scala docs for more

## YakeModel (Keywords Extraction)

Yake is an Unsupervised, Corpus-Independent, Domain and Language-Independent and Single-Document keyword extraction algorithm.

sExtracting keywords from texts has become a challenge for individuals and organizations as the information grows in complexity and size. The need to automate this task so that text can be processed in a timely and adequate manner has led to the emergence of automatic keyword extraction tools. Yake is a novel feature-based system for multi-lingual keyword extraction, which supports texts of different sizes, domain or languages. Unlike other approaches, Yake does not rely on dictionaries nor thesauri, neither is trained against any corpora. Instead, it follows an unsupervised approach which builds upon features extracted from the text, making it thus applicable to documents written in different languages without the need for further knowledge. This can be beneficial for a large number of tasks and a plethora of situations where access to training corpora is either limited or restricted.

The algorithm makes use of the position of a sentence and token. Therefore, to use the annotator, the text should be first sent through a Sentence Boundary Detector and then a tokenizer.

You can tweak the following parameters to get the best result from the annotator.

**Output Annotator Type:** KEYWORD

**Input Annotator Types:** TOKEN

**Reference:** [YakeModel](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/keyword.yake/YakeModel.scala)

**Functions:**

- ***Parameters***

  - `minNGrams: IntParam`: minimum length of a extracted keyword
  - `maxNGrams: IntParam` : maximum length of a extracted keyword
  - `nKeywords: IntParam` :  top N keywords
  - `threshold: FloatParam`: Each keyword will be given a keyword score greater than **0**. Lower the score better the keyword
  - `stopWords: StringArrayParam` : list of stop words
  - `windowSize: IntParam`: Yake will construct a co-occurrence matrix. You can set the window size for the co-occurrence matrix construction from this method. ex: `windowSize=2` will look at two words to both left and right of a candidate word.

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setMinNGrams(int)`: Select the minimum length of a extracted keyword
  - `setMaxNGrams(int)`: Select the maximum length of a extracted keyword
  - `setNKeywords(int)`: Extract the top N keywords
  - `setStopWords(list)`: Set the list of stop words
  - `setThreshold(float)`: Each keyword will be given a keyword score greater than **0**. (Lower the score better the keyword) Set an upper bound for the keyword score from this method.
  - `setWindowSize(int)`: Yake will construct a co-occurrence matrix. You can set the window size for the co-occurrence matrix construction from this method. ex: `windowSize=2` will look at two words to both left and right of a candidate word.

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getBasicStats()`: Calculates basic statistics like total Sentences in the document and assign a tag for each token
  - `getCandidateKeywords()`: Generate candidate keywords
  - `getCoOccurrence()`: Calculate Co-Occurrence for left to right given a window size
  - `getSentences()`: Separate sentences given tokens with sentence metadata

**Example:**

<div class="tabs-box" markdown="1">

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

Refer to the [YakeModel](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.keyword.yake.YakeModel) Scala docs for more

## NER CRF (Named Entity Recognition CRF annotator)

This Named Entity recognition annotator allows for a generic model to be trained by utilizing a CRF machine learning algorithm. Its train data (train_ner) is either a labeled or an [external CoNLL 2003 IOB based](#conll-dataset) spark dataset with Annotations columns. Also the user has to provide [word embeddings annotation](#WordEmbeddings) column.  
Optionally the user can provide an entity dictionary file for better accuracy  

**Output Annotator Type:** Named_Entity  

**Input Annotator Types:** Document, Token, POS, Word_Embeddings  

**Reference:** [NerCrfApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ner/crf/NerCrfApproach.scala) | [NerCrfModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ner/crf/NerCrfModel.scala)

**Functions:**

- ***Parameters***

  - `c0: IntParam`: c0 params defining decay speed for gradient
  - `entities: StringArrayParam` : Entities to recognize
  - `externalFeatures: ExternalResourceParam` : Additional dictionaries to use as a features
  - `includeConfidence: BooleanParam` : Whether to include confidence scores in annotation metadata
  - `l2: DoubleParam`: L2 regularization coefficient
  - `labelColumn: Param[String]` : Column with label per each token
  - `lossEps: DoubleParam`: If Epoch relative improvement less than eps then training is stopped
  - ` maxEpochs: IntParam `: Maximum number of epochs to train
  - `minEpochs: IntParam` : Minimum number of epochs to train
  - `minW: DoubleParam`: Features with less weights then this param value will be filtered
  - `randomSeed: IntParam` : Random seed
  - `verbose: IntParam` : Level of verbosity during training

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setLabelColumn()`: If DatasetPath is not provided, this Seq\[Annotation\] type of column should have labeled data per token
  - `setMinEpochs()`: Minimum number of epochs to train
  - `setMaxEpochs()`: Maximum number of epochs to train
  - `setL2()`: `L2` regularization coefficient for CRF
  - `setC0()`: `c0` defines decay speed for gradient
  - `setLossEps()`: If epoch relative improvement lass than this value, training is stopped
  - `setMinW()`: Features with less weights than this value will be filtered out
  - `setExternalFeatures(path, delimiter, readAs, options)`: Path to file or folder of line separated file that has something like this: Volvo:ORG with such delimiter, readAs `LINE_BY_LINE` or `SPARK_DATASET` with options passed to the latter.
  - `setEntities()`: Array of entities to recognize
  - `setVerbose()`: Verbosity level
  - `setRandomSeed()`: Random seed

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getC0()`: Getter for `c0` params defining decay speed for gradient
  - `getIncludeConfidence()`: Whether or not to calculate prediction confidence by token, includes in metadata
  - `getL2()`: Getter for `L2` regularization coefficient
  - `getLossEps()`: If Epoch relative improvement less than eps then training is stopped
  - `getMaxEpochs()`: Getter for Maximum number of epochs to train
  - `getMinEpochs()`: Getter for Minimum number of epochs to train
  - `getMinW()`: Getter for Features with less weights then this param value will be filtered
  - `getRandomSeed()`: Getter for Random seed
  - `getVerbose()`: Getter for Level of verbosity during training

**Example:**

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
```

</div></div><div class="h3-box" markdown="1">

Refer to the [NerCrfApproach](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfApproach) Scala docs for more details on the API.

## NER DL (Named Entity Recognition Deep Learning annotator)

This Named Entity recognition annotator allows to train generic NER model based on Neural Networks. Its train data (train_ner) is either a labeled or an [external CoNLL 2003 IOB based](#conll-dataset) spark dataset with Annotations columns. Also the user has to provide [word embeddings annotation](#WordEmbeddings) column.  
Neural Network architecture is Char CNNs - BiLSTM - CRF that achieves state-of-the-art in most datasets.

**Output Annotator Type:** Named_Entity

**Input Annotator Types:** Document, Token, Word_Embeddings

**Reference:** [NerDLApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ner/dl/NerDLApproach.scala) | [NerDLModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ner/dl/NerDLModel.scala)

**Functions:**

- ***Parameters***

  - `batchSize: IntParam`: Batch size
  - `configProtoBytes: IntArrayParam` : ConfigProto from tensorflow, serialized into byte array. Get with `config_proto.SerializeToString()`
  - `dropout: FloatParam` : Dropout coefficient
  - `entities: StringArrayParam` : Entities to recognize
  - `evaluationLogExtended: BooleanParam` : Whether logs for validation to be extended: it displays time and evaluation of each label. Default is `false`.
  - `graphFolder: Param[String]` : Folder path that contain external graph files
  - `includeConfidence: BooleanParam` : Whether to include confidence scores in annotation metadata
  - `labelColumn: Param[String]` : Column with label per each token
  - `lr: FloatParam` : Learning Rate
  - `maxEpochs: IntParam` : Maximum number of epochs to train
  - `minEpochs: IntParam` : Minimum number of epochs to train
  - `po: FloatParam` : Learning rate decay coefficient.
  - `randomSeed: IntParam` : Random seed
  - `testDataset: ExternalResourceParam` : Path to test dataset. If set used to calculate statistic on it during training.
  - `useContrib: BooleanParam` : whether to use contrib LSTM Cells. Might slightly improve accuracy.
  - `validationSplit: FloatParam` : Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.
  - `verbose: IntParam` : Level of verbosity during training

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setLabelColumn`: If DatasetPath is not provided, this `Seq\[Annotation\]` type of column should have labeled data per token.
  - `setTestDataset(path: String)`: Set Path to test dataset.
  - `setMaxEpochs`: Maximum number of epochs to train.
  - `setLr`: Initial learning rate.
  - `setPo`: Learning rate decay coefficient. Real Learning Rate: `lr / (1 + po \* epoch)`.
  - `setBatchSize`: Batch size for training.
  - `setDropout`: Dropout coefficient.
  - `setVerbose`: Verbosity level.
  - `setRandomSeed`: Random seed.
  - `setOutputLogsPath`: Folder path to save training logs.

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getBatchSize()`: Getter for Batch size
  - `getConfigProtoBytes()`: ConfigProto from tensorflow, serialized into byte array. Get with `config_proto.SerializeToString()`
  - `getDropout()`: Dropout coefficient
  - `getEnableMemoryOptimizer()`: Memory Optimizer
  - `getEnableOutputLogs()`: Whether to output to annotators log folder
  - `getIncludeConfidence()`: whether to include confidence scores in annotation metadata
  - `getLr()`: Getter for Learning Rate
  - `getMaxEpochs()`: Getter for Maximum number of epochs to train
  - `getMinEpochs()`: Getter for Minimum number of epochs to train
  - `getPo()`: Getter for Learning rate decay coefficient.
  - `getRandomSeed()`: Getter for Random seed
  - `getUseContrib()`: Getter for Whether to use contrib LSTM Cells.
  - `getValidationSplit()`: Getter to choose the proportion of training dataset to be validated against the model on each Epoch.
  - `getVerbose()`: Getter for Level of verbosity during training.

> **Note:** Please check [here](graph.md) in case you get an **IllegalArgumentException** error with a description such as:

    Graph [parameter] should be [value]: Could not find a suitable tensorflow graph for embeddings dim: [value] tags: [value] nChars: [value]. Generate graph by python code in python/tensorflow/ner/create_models before usage and use setGraphFolder Param to point to output.

**Example:**

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
```

</div></div><div class="h3-box" markdown="1">

Refer to the [NerDLApproach](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.ner.dl.NerDLApproach) Scala docs for more details on the API.

## NER Converter (Converts IOB or IOB2 representation of NER to user-friendly)

NER Converter used to finalize work of NER annotators. Combines entites with types `B-`, `I-` and etc. to the Chunks with Named entity in the metadata field (if LightPipeline is used can be extracted after `fullAnnotate()`)

This NER converter can be used to the output of a NER model into the ner chunk format.

**Output Annotator Type:** Chunk

**Input Annotator Types:** Document, Token, Named_Entity

**Reference:** [NerConverter](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ner/NerConverter.scala)

**Functions:**

- ***Parameters***

  - `preservePosition: BooleanParam`: Whether to preserve the original position of the tokens in the original document or use the modified tokens
  - `whiteList: StringArrayParam` : If defined, list of entities to process. The rest will be ignored. Do not include IOB prefix on labels

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setWhiteList(Array(String))`: If defined, list of entities to process. The rest will be ignored. Do not include IOB prefix on labels.
  - `setPreservePosition(Boolean)`: Whether to preserve the original position of the tokens in the original document or use the modified tokens.

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate

**Example:**

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

Refer to the [NerConverter](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.ner.NerConverter) Scala docs for more details on the API.

## Norvig SpellChecker

This annotator retrieves tokens and makes corrections automatically if not found in an English dictionary  

**Output Annotator Type:** Token

**Input Annotator Types:** Token

**Inputs:** Any text for corpus. A list of words for dictionary. A comma separated custom dictionary.

**Train Data:** train_corpus is a spark dataset of text content

**Reference:** [NorvigSweetingApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/norvig/NorvigSweetingApproach.scala) | [NorvigSweetingModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/norvig/NorvigSweetingModel.scala)  

**Functions:**

- ***Parameters***

  - `caseSensitive: BooleanParam` : Sensitivity on spell checking. Defaults to false. Might affect accuracy
  - `dictionary: ExternalResourceParam` : file with a list of correct words
  - `doubleVariants: BooleanParam` : Increase search at cost of performance. Enables extra check for word combinations, More accuracy at performance
  - `dupsLimit: IntParam` : Maximum duplicate of characters in a word to consider.
  - `frequencyPriority: BooleanParam` : Applies frequency over hamming in intersections. When false hamming takes priority
  - `intersections: IntParam` : Hamming intersections to attempt. Defaults to **10**
  - `reductLimit: IntParam` : Word reduction limit. Defaults to **3**
  - `shortCircuit: BooleanParam` : Increase performance at cost of accuracy. Faster but less accurate mode
  - `vowelSwapLimit: IntParam` : Vowel swap attempts. Defaults to **6**
  - `wordSizeIgnore: IntParam` : Minimum size of word before ignoring. Defaults to **3** ,Minimum size of word before moving on. Defaults to **3**.

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setDictionary(path, tokenPattern, readAs, options)`: path to file with properly spelled words, tokenPattern is the regex pattern to identify them in text, readAs `LINE_BY_LINE` or `SPARK_DATASET`, with options passed to Spark reader if the latter is set.
  - `setCaseSensitive(boolean)`: defaults to `false`. Might affect accuracy
  - `setDoubleVariants(boolean)`: enables extra check for word combinations, more accuracy at performance
  - `setShortCircuit(boolean)`: faster but less accurate mode
  - `setWordSizeIgnore(int)`: Minimum size of word before moving on. Defaults to **3**.
  - `setDupsLimit(int)`: Maximum duplicate of characters to account for. Defaults to **2**.
  - `setReductLimit(int)`: Word reduction limit. Defaults to **3**
  - `setIntersections(int)`: Hamming intersections to attempt. Defaults to **10**.
  - `setVowelSwapLimit(int)`: Vowel swap attempts. Defaults to **6**.

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getCaseSensitive()`: Sensitivity on spell checking. Defaults to `false`. Might affect accuracy
  - `getDoubleVariants()`: Increase search at cost of performance. Enables extra check for word combinations
  - `getDupsLimit()`: Maximum duplicate of characters in a word to consider. Defaults to **2** .Maximum duplicate of characters to account for. Defaults to **2**.
  - `getFrequencyPriority()`: Applies frequency over hamming in intersections. When false hamming takes priority
  - `getIntersections()`: Hamming intersections to attempt. Defaults to **10**
  - `getReductLimit()`: Word reduction limit. Defaults to **3**
  - `getShortCircuit()`: Increase performance at cost of accuracy. Faster but less accurate mode
  - `getVowelSwapLimit()`: Vowel swap attempts. Defaults to **6**
  - `getWordSizeIgnore()`: Minimum size of word before ignoring. Defaults to **3**, Minimum size of word before moving on. Defaults to **3**.

**Example:**

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

Refer to the [NorvigSweetingApproach](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingApproach) Scala docs for more details on the API.

## Symmetric SpellChecker

This spell checker is inspired on Symmetric Delete algorithm. It retrieves tokens and utilizes distance metrics to compute possible derived words  

**Output Annotator Type:** Token  

**Input Annotator Types:** Token

**Inputs:** Any text for corpus. A list of words for dictionary. A comma separated custom dictionary.

**Train Data:** train_corpus is a spark dataset of text content

**Reference:** [SymmetricDeleteApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/symmetric/SymmetricDeleteApproach.scala) | [SymmetricDeleteModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/symmetric/SymmetricDeleteModel.scala)  

**Functions:**

- ***Parameters***

  - `deletesThreshold: IntParam` : minimum frequency of corrections a word needs to have to be considered from training. Increase if training set is LARGE.
  - `dictionary: ExternalResourceParam` : file with a list of correct words
  - `dupsLimit: IntParam` : maximum duplicate of characters in a word to consider
  - `frequencyThreshold: IntParam` : minimum frequency of words to be considered from training. Increase if training set is LARGE
  - `longestWordLength: IntParam` : length of longest word in corpus
  - `maxEditDistance: IntParam` : max edit distance characters to derive strings from a word
  - `maxFrequency: LongParam`: maximum frequency of a word in the corpus
  - `minFrequency: LongParam` : minimum frequency of a word in the corpus

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setDeletesThreshold(value: Int)`: minimum frequency of corrections a word needs to have to be considered from training. Increase if training set is LARGE. Defaults to **0**.
  - `setFrequencyThreshold(value: Int)`: minimum frequency of words to be considered from training. Increase if training set is LARGE. Defaults to **0**.
  - `setLongestWordLength(value: Int)`: length of longest word in corpus
  - `setMaxFrequency(value: Long)`: maximum frequency of a word in the corpus
  - `setMinFrequency(value: Long)`: minimum frequency of a word in the corpus
  - `setDictionary(path, tokenPattern, readAs, options)`: Optional dictionary of properly written words. If provided, significantly boosts spell checking performance
  - `setMaxEditDistance(distance)`: Maximum edit distance to calculate possible derived words. Defaults to **3**.

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getDeletesThreshold()`: minimum frequency of corrections a word needs to have to be considered from training. Increase if training set is `LARGE`. Defaults to **0**
  - `getDupsLimit()`: maximum duplicate of characters in a word to consider. Defaults to **2**
  - `getFrequencyThreshold()`: minimum frequency of words to be considered from training. Increase if training set is `LARGE`. Defaults to **0**.
  - `getMaxEditDistance()`: max edit distance characters to derive strings from a word.

**Example:**

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
spell_checker = SymmetricDeleteApproach() \
    .setInputCols(["token"]) \
    .setOutputCol("spell")
```

```scala
val spellChecker = new SymmetricDeleteApproach()
    .setInputCols(Array("normalized"))
    .setOutputCol("spell")
```  

</div></div><div class="h3-box" markdown="1">

Refer to the [SymmetricDeleteApproach](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteApproach) Scala docs for more details on the API.

## Context SpellChecker

Implements Noisy Channel Model Spell Algorithm. Correction candidates are extracted combining context information and word information  

**Output Annotator Type:** Token  

**Input Annotator Types:** Token  

**Inputs:** Any text for corpus. A list of words for dictionary. A comma separated custom dictionary.

**Train Data:** train_corpus is a spark dataset of text content

**Reference:** [ContextSpellCheckerApproach](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/context/ContextSpellCheckerApproach.scala) | [ContextSpellCheckerModel](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/context/ContextSpellCheckerModel.scala)  

**Functions:**

- ***Parameters***

  - `languageModelClasses: IntParam`: Number of classes to use during factorization of the softmax output in the LM.
  - `wordMaxDistance :IntParam`: Maximum distance for the generated candidates for every word.
  - `maxCandidates :IntParam`: Maximum number of candidates for every word.
  - `caseStrategy :IntParam`: What case combinations to try when generating candidates. `ALL_UPPER_CASE = 0, FIRST_LETTER_CAPITALIZED = 1, ALL = 2`.
  - `errorThreshold :Float`: Threshold perplexity for a word to be considered as an error.
  - `tradeoff :Float`: Tradeoff between the cost of a word error and a transition in the language model.
  - `maxWindowLen :IntParam`: Maximum size for the window used to remember history prior to every correction.
  - `gamma: Float`: Controls the influence of individual word frequency in the decision.

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setLanguageModelClasses(languageModelClasses:Int)`: Number of classes to use during factorization of the softmax output in the LM. Defaults to **2000**.
  - `setWordMaxDistance(dist:Int)`: Maximum distance for the generated candidates for every word. Defaults to **3**.
  - `setMaxCandidates(candidates:Int)`: Maximum number of candidates for every word. Defaults to **6**.
  - `setCaseStrategy(strategy:Int)`: What case combinations to try when generating candidates. `ALL_UPPER_CASE = 0, FIRST_LETTER_CAPITALIZED = 1, ALL = 2`. Defaults to **2**.
  - `setErrorThreshold(threshold:Float)`: Threshold perplexity for a word to be considered as an error. Defaults to **10f**.
  - `setTradeoff(alpha:Float)`: Tradeoff between the cost of a word error and a transition in the language model. Defaults to **18.0f**.
  - `setMaxWindowLen(length:Integer)`: Maximum size for the window used to remember history prior to every correction. Defaults to **5**.
  - `setGamma(g:Float)`: Controls the influence of individual word frequency in the decision.
  - `updateVocabClass(label:String, vocab:Array(String), append:boolean)`: Update existing vocabulary classes so they can cover new words. If append set to `false` overwrite vocabulary class in the model by new words, if `true` extends existing vocabulary class. Defaults to `true`.  
  - `updateRegexClass(label:String, regex:String)`: Update existing regex rule for the class defined by regex.

- Train:

  - `setWeightedDistPath(weightedDistPath:String)`: The path to the file containing the weights for the levenshtein distance.
  - `setEpochs(epochs:Int)`: Number of epochs to train the language model. Defaults to **2**.
  - `setInitialBatchSize(batchSize:Int)`: Batch size for the training in NLM. Defaults to **24**.
  - `setInitialRate(initialRate:Float)`: Initial learning rate for the LM. Defaults to **.7f**.
  - `setFinalRate(finalRate:Float)`: Final learning rate for the LM. Defaults to **0.0005f**.
  - `setValidationFraction(validationFraction:Float)`: Percentage of datapoints to use for validation. Defaults to **.1f**.
  - `setMinCount(minCount:Float)`: Min number of times a token should appear to be included in vocab. Defaults to **3.0f**.
  - `setCompoundCount(compoundCount:Int)`: Min number of times a compound word should appear to be included in vocab. Defaults to **5**.
  - `setClassCount(classCount:Int)`: Min number of times the word need to appear in corpus to not be considered of a special class. Defaults to **15**.

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate

**Example:**

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
spell_checker = ContextSpellCheckerApproach() \
    .setInputCols(["token"]) \
    .setOutputCol("spell") \
    .fit(train_corpus) \
    .setErrorThreshold(4.0)\
    .setTradeoff(6.0)
```

```scala
val spellChecker = new ContextSpellCheckerApproach()
    .setInputCols(Array("token"))
    .setOutputCol("spell")
    .fit(trainCorpus)
    .setErrorThreshold(4.0)
    .setTradeoff(6.0)
```

</div></div><div class="h3-box" markdown="1">

Refer to the [ContextSpellCheckerApproach](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.spell.context.ContextSpellCheckerApproach) Scala docs for more details on the API.

## Dependency Parsers

Dependency parser provides information about word relationship. For example, dependency parsing can tell you what the subjects and objects of a verb are, as well as which words are modifying (describing) the subject. This can help you find precise answers to specific questions.
The following diagram illustrates a dependency-style analysis using the standard graphical method favored in the dependency-parsing community.

![Dependency Parser](\assets\images\dependency_parser.png)

Relations among the words are illustrated above the sentence with directed, labeled arcs from heads to dependents. We call this a typed dependency structure because the labels are drawn from a fixed inventory of grammatical relations. It also includes a root node that explicitly marks the root of the tree, the head of the entire structure. [1]

</div><div class="h3-box" markdown="1">

## Untyped Dependency Parser (Unlabeled grammatical relation)

Unlabeled parser that finds a grammatical relation between two words in a sentence. Its input is a directory with dependency treebank files.  

**Output Annotator Type:** Dependency  

**Input Annotator Types:** Document, POS, Token  

**Reference:** [DependencyParserApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/parser/dep/DependencyParserApproach.scala) | [DependencyParserModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/parser/dep/DependencyParserModel.scala)  

**Functions:**

- ***Parameters***

  - `conllU: ExternalResourceParam` : Universal Dependencies source files
  - `dependencyTreeBank: ExternalResourceParam`: Dependency treebank source files
  - `numberOfIterations: IntParam`: Number of iterations in training, converges to better accuracy

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setNumberOfIterations(int)`: Number of iterations in training, converges to better accuracy
  - `setDependencyTreeBank(String)`: Dependency treebank folder with files in [Penn Treebank format](http://www.nltk.org/nltk_data/)
  - `setConllU(String)`: Path to a file in [CoNLL-U format](https://universaldependencies.org/format.html)

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate
  - `getNumberOfIterations()`: Number of iterations in training, converges to better accuracy
  - `getFilesContentTreeBank()`: Gets a iterable TreeBank
  - `getTrainingSentences()`: Gets a list of ConnlU training sentences

**Example:**

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

Refer to the [DependencyParserApproach](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserApproach) Scala docs for more details on the API.

## Typed Dependency Parser (Labeled grammatical relation)

Labeled parser that finds a grammatical relation between two words in a sentence. Its input is a CoNLL2009 or ConllU dataset.  

**Output Annotator Type:** Labeled Dependency  

**Input Annotator Types:** Token, POS, Dependency  

**Reference:** [TypedDependencyParserApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/parser/typdep/TypedDependencyParserApproach.scala) | [TypedDependencyParserModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/parser/typdep/TypedDependencyParserModel.scala)  

**Functions:**

- ***Parameters***

  - `conllU: ExternalResourceParam` : Universal Dependencies source files
  - ` conll2009: ExternalResourceParam `: Path to file with CoNLL 2009 format
  - `numberOfIterations: IntParam`: Number of iterations in training, converges to better accuracy

- ***Parameter Setters***

  - `setInputCol(String)`: Sets required input annotator types
  - `setOutputCol(String)`: Sets expected output annotator types
  - `setNumberOfIterations(int)`: Number of iterations in training, converges to better accuracy
  - `setConll2009(String)`: Path to a file in [CoNLL 2009 format](https://ufal.mff.cuni.cz/conll2009-st/trial-data.html)
  - `setConllU(String)`: Path to a file in [CoNLL-U format](https://universaldependencies.org/format.html)

- ***Parameter Getters***

  - `getInputCols()`: Input annotations columns currently used
  - `getOutputCols()`: Gets annotation column name going to generate

**Example:**

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

Refer to the [TypedDependencyParserApproach](https://nlp.johnsnowlabs.com/api/index#com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserApproach) Scala docs for more details on the API.

## References

[1] Speech and Language Processing. Daniel Jurafsky & James H. Martin. 2018
