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

**API:** [Tokenizer](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/Tokenizer) |

**Source:** [Tokenizer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/Tokenizer.scala) |

</div></div><div class="h3-box" markdown="1">

## DocumentNormalizer (Text cleaning)

Annotator which normalizes raw text from tagged text, e.g. scraped web pages or xml documents, from document type columns into Sentence.  

**Output Annotator Type:** Document  

**Input Annotator Types:** Document  

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

**API:** [DocumentNormalizer](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/DocumentNormalizer) |

**Source:** [DocumentNormalizer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/DocumentNormalizer.scala) |

</div></div><div class="h3-box" markdown="1">

## Normalizer (Text cleaning)

Removes all dirty characters from text following a regex pattern and transforms words based on a provided dictionary  

**Output Annotator Type:** Token  

**Input Annotator Types:** Token  

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

**API:** [Normalizer](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/Normalizer) |

**Source:** [Normalizer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/Normalizer.scala) |

</div></div><div class="h3-box" markdown="1">

## Stemmer

Returns hard-stems out of words with the objective of retrieving the meaningful part of the word

**Output Annotator Type:** Token  

**Input Annotator Types:** Token  

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

**API:** [Stemmer](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/Stemmer) |

**Source:** [Stemmer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/Stemmer.scala) |

</div></div><div class="h3-box" markdown="1">

## Lemmatizer

Retrieves lemmas out of words with the objective of returning a base dictionary word  

**Output Annotator Type:** Token  

**Input Annotator Types:** Token

<!-- **Input**: abduct -> abducted abducting abduct abducts -->

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

**API:** [Lemmatizer](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/Lemmatizer) |

**Source:** [Lemmatizer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/Lemmatizer.scala) |

</div></div><div class="h3-box" markdown="1">

## StopWordsCleaner

This annotator excludes from a sequence of strings (e.g. the output of a `Tokenizer()`, `Normalizer()`, `Lemmatizer()`, and `Stemmer()`) and drops all the stop words from the input sequences.

**Output Annotator Type:** token

**Input Annotator Types:** token

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

**API:** [StopWordsCleaner](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/StopWordsCleaner) |

**Source:** [StopWordsCleaner](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/StopWordsCleaner.scala) |

</div></div><div class="h3-box" markdown="1">

## RegexMatcher

Uses a reference file to match a set of regular expressions and put them inside a provided key. File must be comma separated.  

**Output Annotator Type:** Regex  

**Input Annotator Types:** Document  

<!-- **Input:** `the\\s\\w+`, "followed by 'the'"   -->

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

**API:** [RegexMatcher](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/RegexMatcher) |

**Source:** [RegexMatcher](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/RegexMatcher.scala) | [RegexMatcherModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/RegexMatcherModel.scala)  

</div></div><div class="h3-box" markdown="1">

## TextMatcher (Phrase matching)

Annotator to match entire phrases (by token) provided in a file against a Document  

**Output Annotator Type:** Entity  

**Input Annotator Types:** Document, Token

<!-- **Input**: hello world, I am looking for you -->

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

**API:** [TextMatcher](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/TextMatcher) |

**Source:** [TextMatcher](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/TextMatcher.scala) | [TextMatcherModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/TextMatcherModel.scala)  

</div></div><div class="h3-box" markdown="1">

## Chunker

This annotator matches a pattern of part-of-speech tags in order to return meaningful phrases from document

**Output Annotator Type:** Chunk  

**Input Annotator Types:** Document, POS  

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

**API:** [Chunker](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/Chunker) |

**Source:** [Chunker](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/Chunker.scala) |

</div></div><div class="h3-box" markdown="1">

## NGramGenerator

`NGramGenerator` annotator takes as input a sequence of strings (e.g. the output of a `Tokenizer`, `Normalizer`, `Stemmer`, `Lemmatizer`, and `StopWordsCleaner`). The parameter `n` is used to determine the number of terms in each n-gram. The output will consist of a sequence of n-grams where each n-gram is represented by a space-delimited string of n consecutive words with annotatorType `CHUNK` same as the `Chunker` annotator.

**Output Annotator Type:** CHUNK  

**Input Annotator Types:** TOKEN  

**Reference:** [NGramGenerator](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/NGramGenerator.scala)  

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

## DateMatcher

Reads from different forms of date and time expressions and converts them to a provided date format. Extracts only ONE date per sentence. Use with sentence detector for more matches.

**Output Annotator Type:** Date  

**Input Annotator Types:** Document  

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

**API:** [DateMatcher](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/DateMatcher) |

**Source:** [DateMatcher](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/DateMatcher.scala) |

</div></div><div class="h3-box" markdown="1">

## MultiDateMatcher

Reads from multiple different forms of date and time expressions and converts them to a provided date format. Extracts multiple dates per sentence.

**Output Annotator Type:** Date  

**Input Annotator Types:** Document  

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

**API:** [MultiDateMatcher](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/MultiDateMatcher) |

**Source:** [MultiDateMatcher](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/MultiDateMatcher.scala) |

</div></div><div class="h3-box" markdown="1">

## SentenceDetector

Finds sentence bounds in raw text. Applies rules from Pragmatic Segmenter.  

**Output Annotator Type:** Sentence

**Input Annotator Types:** Document  

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

**API:** [SentenceDetector](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/sbd/pragmatic/SentenceDetector) |

**Source:** [SentenceDetector](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/sbd/pragmatic/SentenceDetector.scala)  

</div></div><div class="h3-box" markdown="1">

## POSTagger (Part of speech tagger)

Sets a POS tag to each word within a sentence. Its train data (train_pos) is a spark dataset of [POS format values](#TrainPOS) with Annotation columns.  

**Output Annotator Type:** POS  

**Input Annotator Types:** Document, Token  

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

**API:** [PerceptronApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/pos/perceptron/PerceptronApproach) |

**Source:** [PerceptronApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/pos/perceptron/PerceptronApproach.scala) | [PerceptronModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/pos/perceptron/PerceptronModel.scala)  

</div></div><div class="h3-box" markdown="1">

## ViveknSentimentDetector

Scores a sentence for a sentiment
  
**Output Annotator Type:** sentiment  

**Input Annotator Types:** Document, Token  

<!-- **Input:** File or folder of text files of positive and negative data   -->
  
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

**API:** [ViveknSentimentApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/sda/vivekn/ViveknSentimentApproach) |

**Source:** [ViveknSentimentApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/sda/vivekn/ViveknSentimentApproach.scala) | [ViveknSentimentModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/sda/vivekn/ViveknSentimentModel.scala)

</div></div><div class="h3-box" markdown="1">

## SentimentDetector (Sentiment analysis)

Scores a sentence for a sentiment  

**Output Annotator Type:** Sentiment  

**Input Annotator Types:** Document, Token  

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

**API:** [SentimentDetector](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/sda/pragmatic/SentimentDetector) |

**Reference:** [SentimentDetector](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/sda/pragmatic/SentimentDetector.scala) | [SentimentDetectorModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/sda/pragmatic/SentimentDetectorModel.scala)  

</div></div><div class="h3-box" markdown="1">

## WordEmbeddings

Word Embeddings lookup annotator that maps tokens to vectors  

**Output Annotator Type:** Word_Embeddings  

**Input Annotator Types:** Document, Token  

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

**API:** [WordEmbeddings](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/embeddings/WordEmbeddings) |

**Source:**  [WordEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/WordEmbeddings.scala) | [WordEmbeddingsModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/WordEmbeddingsModel.scala)  

</div><div class="h3-box" markdown="1">

## BertEmbeddings

BERT (Bidirectional Encoder Representations from Transformers) provides dense vector representations for natural language by using a deep, pre-trained neural network with the Transformer architecture

You can find the pre-trained models for `BertEmbeddings` in the [Spark NLP Models](https://github.com/JohnSnowLabs/spark-nlp-models) repository

**Output Annotator Type:** Word_Embeddings  

**Input Annotator Types:** Document, Token

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

**API:** [BertEmbeddings](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/embeddings/BertEmbeddings) |

**Source:** [BertEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/BertEmbeddings.scala) |

</div><div class="h3-box" markdown="1">

## BertSentenceEmbeddings

BERT (Bidirectional Encoder Representations from Transformers) provides dense vector representations for natural language by using a deep, pre-trained neural network with the Transformer architecture

You can find the pre-trained models for `BertEmbeddings` in the [Spark NLP Models](https://github.com/JohnSnowLabs/spark-nlp-models) repository

**Output Annotator Type:** Sentence_Embeddings  

**Input Annotator Types:** Document

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

**API:** [BertSentenceEmbeddings](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/embeddings/BertSentenceEmbeddings) |

**Source:** [BertSentenceEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/BertSentenceEmbeddings.scala) |

</div></div><div class="h3-box" markdown="1">

## ElmoEmbeddings

Computes contextualized word representations using character-based word representations and bidirectional LSTMs

You can find the pre-trained model for `ElmoEmbeddings` in the  [Spark NLP Models](https://github.com/JohnSnowLabs/spark-nlp-models#english---models) repository

**Output Annotator Type:** Word_Embeddings

**Input Annotator Types:** Document, Token

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

**API:** [ElmoEmbeddings](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/embeddings/ElmoEmbeddings) |

**Source:** [ElmoEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/ElmoEmbeddings.scala) |

</div></div><div class="h3-box" markdown="1">

## AlbertEmbeddings

Computes contextualized word representations using "A Lite" implementation of BERT algorithm by applying parameter-reduction techniques

You can find the pre-trained model for `AlbertEmbeddings` in the  [Spark NLP Models](https://github.com/JohnSnowLabs/spark-nlp-models#english---models) repository

**Output Annotator Type:** Word_Embeddings

**Input Annotator Types:** Document, Token

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

**API:** [AlbertEmbeddings](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/embeddings/AlbertEmbeddings) |

**Source:** [AlbertEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/AlbertEmbeddings.scala) |

</div></div><div class="h3-box" markdown="1">

## XlnetEmbeddings

Computes contextualized word representations using combination of Autoregressive Language Model and Permutation Language Model

You can find the pre-trained model for `XlnetEmbeddings` in the  [Spark NLP Models](https://github.com/JohnSnowLabs/spark-nlp-models#english---models) repository

**Output Annotator Type:** Word_Embeddings

**Input Annotator Types:** Document, Token

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

**API:** [XlnetEmbeddings](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/embeddings/XlnetEmbeddings) |

**Source:** [XlnetEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/XlnetEmbeddings.scala) |

</div></div><div class="h3-box" markdown="1">

## UniversalSentenceEncoder

The Universal Sentence Encoder encodes text into high dimensional vectors that can be used for text classification, semantic similarity, clustering and other natural language tasks.

**Output Annotator Type:** SENTENCE_EMBEDDINGS

**Input Annotator Types:** Document

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

**API:** [UniversalSentenceEncoder](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/embeddings/UniversalSentenceEncoder) |

**Source:** [UniversalSentenceEncoder](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/UniversalSentenceEncoder.scala) |

</div></div><div class="h3-box" markdown="1">

## SentenceEmbeddings

This annotator converts the results from `WordEmbeddings`, `BertEmbeddings`, `ElmoEmbeddings`, `AlbertEmbeddings`, or `XlnetEmbeddings` into `sentence` or `document` embeddings by either summing up or averaging all the word embeddings in a sentence or a document (depending on the `inputCols`).

**Output Annotator Type:** SENTENCE_EMBEDDINGS

**Input Annotator Types:** Document

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

**API:** [SentenceEmbeddings](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/embeddings/SentenceEmbeddings) |

**Source:** [SentenceEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/SentenceEmbeddings.scala) |

</div></div><div class="h3-box" markdown="1">

## ChunkEmbeddings

This annotator utilizes `WordEmbeddings` or `BertEmbeddings` to generate chunk embeddings from either `Chunker`, `NGramGenerator`, or `NerConverter` outputs.

**Output Annotator Type:** CHUNK

**Input Annotator Types:** CHUNK, Word_Embeddings

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

**API:** [ChunkEmbeddings](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/embeddings/ChunkEmbeddings) |

**Source:** [ChunkEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/ChunkEmbeddings.scala) |

</div></div><div class="h3-box" markdown="1">

## ClassifierDL (Multi-class Text Classification)

ClassifierDL is a generic Multi-class Text Classification. ClassifierDL uses the state-of-the-art Universal Sentence Encoder as an input for text classifications. The ClassifierDL annotator uses a deep learning model (DNNs) we have built inside TensorFlow and supports up to 100 classes

**Output Annotator Type:** CATEGORY

**Input Annotator Types:** SENTENCE_EMBEDDINGS

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

**API:** [ClassifierDLApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/classifier/dl/ClassifierDLApproach) |

**Source:** [ClassifierDLApproach](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/ClassifierDLApproach.scala) | [ClassifierDLModel](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/ClassifierDLModel.scala)

</div><div class="h3-box" markdown="1">

## MultiClassifierDL (Multi-label Text Classification)

 MultiClassifierDL is a Multi-label Text Classification. MultiClassifierDL uses a Bidirectional GRU with Convolution model that we have built inside TensorFlow and supports up to 100 classes. The input to MultiClassifierDL is Sentence Embeddings such as state-of-the-art UniversalSentenceEncoder, BertSentenceEmbeddings, or SentenceEmbeddings

**Output Annotator Type:** CATEGORY

**Input Annotator Types:** SENTENCE_EMBEDDINGS

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

**API:** [MultiClassifierDLApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/classifier/dl/MultiClassifierDLApproach) |

**Source:** [MultiClassifierDLApproach](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/MultiClassifierDLApproach.scala) | [MultiClassifierDLModel](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/MultiClassifierDLModel.scala)

</div><div class="h3-box" markdown="1">

## SentimentDL (Multi-class Sentiment Analysis annotator)

SentimentDL is an annotator for multi-class sentiment analysis. This annotator comes with 2 available pre-trained models trained on IMDB and Twitter datasets

**Output Annotator Type:** CATEGORY

**Input Annotator Types:** SENTENCE_EMBEDDINGS

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

**API:** [SentimentDLApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/classifier/dl/SentimentDLApproach) |

**Source:** [SentimentDLApproach](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/SentimentDLApproach.scala) | [SentimentDLModel](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/SentimentDLModel.scala)

</div><div class="h3-box" markdown="1">

## LanguageDetectorDL (Language Detection and Identiffication)

LanguageDetectorDL is a state-of-the-art language detection and identification annotator trained by using TensorFlow/keras neural networks.

**Output Annotator Type:** LANGUAGE

**Input Annotator Types:** DOCUMENT or SENTENCE

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

**API:** [LanguageDetectorDL](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/ld/dl/LanguageDetectorDL) |

**Source:** [LanguageDetectorDL](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ld/dl/LanguageDetectorDL.scala) |

</div><div class="h3-box" markdown="1">

## YakeModel (Keywords Extraction)

Yake is an Unsupervised, Corpus-Independent, Domain and Language-Independent and Single-Document keyword extraction algorithm.

sExtracting keywords from texts has become a challenge for individuals and organizations as the information grows in complexity and size. The need to automate this task so that text can be processed in a timely and adequate manner has led to the emergence of automatic keyword extraction tools. Yake is a novel feature-based system for multi-lingual keyword extraction, which supports texts of different sizes, domain or languages. Unlike other approaches, Yake does not rely on dictionaries nor thesauri, neither is trained against any corpora. Instead, it follows an unsupervised approach which builds upon features extracted from the text, making it thus applicable to documents written in different languages without the need for further knowledge. This can be beneficial for a large number of tasks and a plethora of situations where access to training corpora is either limited or restricted.

The algorithm makes use of the position of a sentence and token. Therefore, to use the annotator, the text should be first sent through a Sentence Boundary Detector and then a tokenizer.

You can tweak the following parameters to get the best result from the annotator.

**Output Annotator Type:** KEYWORD

**Input Annotator Types:** TOKEN

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

**API:** [YakeModel](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/keyword.yake/YakeModel) |

**Source:** [YakeModel](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/keyword.yake/YakeModel.scala) |

</div><div class="h3-box" markdown="1">

## NER CRF (Named Entity Recognition CRF annotator)

This Named Entity recognition annotator allows for a generic model to be trained by utilizing a CRF machine learning algorithm. Its train data (train_ner) is either a labeled or an [external CoNLL 2003 IOB based](#conll-dataset) spark dataset with Annotations columns. Also the user has to provide [word embeddings annotation](#WordEmbeddings) column.  
Optionally the user can provide an entity dictionary file for better accuracy  

**Output Annotator Type:** Named_Entity  

**Input Annotator Types:** Document, Token, POS, Word_Embeddings  

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

**API:** [NerCrfApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/ner/crf/NerCrfApproach) |

**Source:** [NerCrfApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ner/crf/NerCrfApproach.scala) | [NerCrfModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ner/crf/NerCrfModel.scala)

</div></div><div class="h3-box" markdown="1">

## NER DL (Named Entity Recognition Deep Learning annotator)

This Named Entity recognition annotator allows to train generic NER model based on Neural Networks. Its train data (train_ner) is either a labeled or an [external CoNLL 2003 IOB based](#conll-dataset) spark dataset with Annotations columns. Also the user has to provide [word embeddings annotation](#WordEmbeddings) column.  
Neural Network architecture is Char CNNs - BiLSTM - CRF that achieves state-of-the-art in most datasets.

**Output Annotator Type:** Named_Entity

**Input Annotator Types:** Document, Token, Word_Embeddings

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

**API:** [NerDLApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/ner/dl/NerDLApproach) |

**Source:** [NerDLApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ner/dl/NerDLApproach.scala) | [NerDLModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ner/dl/NerDLModel.scala)

</div></div><div class="h3-box" markdown="1">

## NER Converter (Converts IOB or IOB2 representation of NER to user-friendly)

NER Converter used to finalize work of NER annotators. Combines entites with types `B-`, `I-` and etc. to the Chunks with Named entity in the metadata field (if LightPipeline is used can be extracted after `fullAnnotate()`)

This NER converter can be used to the output of a NER model into the ner chunk format.

**Output Annotator Type:** Chunk

**Input Annotator Types:** Document, Token, Named_Entity

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

**API:** [NerConverter](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/ner/NerConverter) |

**Source:** [NerConverter](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ner/NerConverter.scala) |

</div></div><div class="h3-box" markdown="1">

## Norvig SpellChecker

This annotator retrieves tokens and makes corrections automatically if not found in an English dictionary  

**Output Annotator Type:** Token

**Input Annotator Types:** Token

**Inputs:** Any text for corpus. A list of words for dictionary. A comma separated custom dictionary.

**Train Data:** train_corpus is a spark dataset of text content

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

**API:** [NorvigSweetingApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/spell/norvig/NorvigSweetingApproach) |

**Source:** [NorvigSweetingApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/norvig/NorvigSweetingApproach.scala) | [NorvigSweetingModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/norvig/NorvigSweetingModel.scala)  

</div></div><div class="h3-box" markdown="1">

## Symmetric SpellChecker

This spell checker is inspired on Symmetric Delete algorithm. It retrieves tokens and utilizes distance metrics to compute possible derived words  

**Output Annotator Type:** Token  

**Input Annotator Types:** Token

**Inputs:** Any text for corpus. A list of words for dictionary. A comma separated custom dictionary.

**Train Data:** train_corpus is a spark dataset of text content

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

**API:** [SymmetricDeleteApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/spell/symmetric/SymmetricDeleteApproach) |

**Source:** [SymmetricDeleteApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/symmetric/SymmetricDeleteApproach.scala) | [SymmetricDeleteModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/symmetric/SymmetricDeleteModel.scala)  

</div></div><div class="h3-box" markdown="1">

## Context SpellChecker

Implements Noisy Channel Model Spell Algorithm. Correction candidates are extracted combining context information and word information  

**Output Annotator Type:** Token  

**Input Annotator Types:** Token  

**Inputs:** Any text for corpus. A list of words for dictionary. A comma separated custom dictionary.

**Train Data:** train_corpus is a spark dataset of text content

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

**API:** [ContextSpellCheckerApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/spell/context/ContextSpellCheckerApproach) |

**Source:** [ContextSpellCheckerApproach](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/context/ContextSpellCheckerApproach.scala) | [ContextSpellCheckerModel](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/context/ContextSpellCheckerModel.scala)  

</div></div><div class="h3-box" markdown="1">

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

**API:** [DependencyParserApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/parser/dep/DependencyParserApproach) |

**Source:** [DependencyParserApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/parser/dep/DependencyParserApproach.scala) | [DependencyParserModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/parser/dep/DependencyParserModel.scala)  

</div></div><div class="h3-box" markdown="1">

## Typed Dependency Parser (Labeled grammatical relation)

Labeled parser that finds a grammatical relation between two words in a sentence. Its input is a CoNLL2009 or ConllU dataset.  

**Output Annotator Type:** Labeled Dependency  

**Input Annotator Types:** Token, POS, Dependency  

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

**API:** [TypedDependencyParserApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/parser/typdep/TypedDependencyParserApproach) |

**Source:** [TypedDependencyParserApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/parser/typdep/TypedDependencyParserApproach.scala) | [TypedDependencyParserModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/parser/typdep/TypedDependencyParserModel.scala)  

</div></div><div class="h3-box" markdown="1">

## References

[1] Speech and Language Processing. Daniel Jurafsky & James H. Martin. 2018
