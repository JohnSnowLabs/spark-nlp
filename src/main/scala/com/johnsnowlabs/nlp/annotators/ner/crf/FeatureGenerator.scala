package com.johnsnowlabs.nlp.annotators.ner.crf

import com.johnsnowlabs.ml.crf._
import com.johnsnowlabs.nlp.annotators.common.{TaggedSentence, WordpieceEmbeddingsSentence}
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsRetriever

import scala.collection.mutable


/**
  * Generates features for CrfBasedNer
  */
case class FeatureGenerator(dictFeatures: DictionaryFeatures) {

  val shapeEncoding = Map(
    '.' -> '.', ',' -> '.',
    ':' -> ':', ';' -> ':', '?' -> ':', '!' -> ':',
    '-' -> '-', '+' -> '-', '*' -> '-', '/' -> '-', '=' -> '-', '|' -> '-', '_' -> '-', '%' -> '-',
    '(' -> '(', '{' -> '(', '[' -> '(', '<' -> '(',
    ')' -> ')', '}' -> ')', ']' -> ')', '>' -> ')'
  )

  def getShape(token: String) = {
    token.map(c =>{
      if (c.isLower)
        'L'
      else if (c.isUpper)
        'U'
      else if (c.isDigit)
        'D'
      shapeEncoding.getOrElse(c, c)
    })
  }

  def shrink(str: String) = {
    val builder = new StringBuilder()
    for (c <- str) {
      if (builder.length == 0 || builder.last != c)
        builder.append(c)
    }
    builder.toString
  }

  object TokenType extends Enumeration {
    type TokenType = Value
    val AllUpper = Value(1 << 0)
    val AllDigit = Value(1 << 1)
    val AllSymbol = Value(1 << 2)
    val AllUpperDigit = Value(1 << 3)
    val AllUpperSymbol = Value(1 << 4)
    val AllDigitSymbol = Value(1 << 5)
    val AllUpperDigitSymbol = Value(1 << 6)
    val StartsUpper = Value(1 << 7)
    val AllLetter = Value(1 << 8)
    val AllAlnum = Value(1 << 9)

    val allTypes = values.max.id * 2 - 1
  }

  val digitDelims = Seq(',', '.')

  def getType(token: String): Int = {
    var types = TokenType.allTypes

    def remove(t: TokenType.TokenType) = {
      types = types & (~t.id)
    }

    var isFirst = true
    for (c <- token) {
      if (c.isUpper) {
        remove(TokenType.AllDigit)
        remove(TokenType.AllSymbol)
        remove(TokenType.AllDigitSymbol)
      }
      else if (c.isDigit || digitDelims.contains(c)) {
        remove(TokenType.AllUpper)
        remove(TokenType.AllSymbol)
        remove(TokenType.AllUpperSymbol)
        remove(TokenType.AllLetter)
      }
      else if (c.isLower) {
        remove(TokenType.AllUpper)
        remove(TokenType.AllDigit)
        remove(TokenType.AllSymbol)
        remove(TokenType.AllUpperDigit)
        remove(TokenType.AllUpperSymbol)
        remove(TokenType.AllDigitSymbol)
        remove(TokenType.AllUpperDigitSymbol)
      }
      else {
        remove(TokenType.AllUpper)
        remove(TokenType.AllDigit)
        remove(TokenType.AllUpperDigit)
        remove(TokenType.AllLetter)
        remove(TokenType.AllAlnum)
      }

      if (isFirst && !c.isUpper)
        remove(TokenType.StartsUpper)

      isFirst = false
    }

    val result = TokenType.values
      .filter(value => (value.id & types) > 0)
      .map(value => value.id)
      .headOption

    result.getOrElse(0)
  }

  def isDigitOrPredicate(token: String, predicate: Function[Char, Boolean]) = {
    var hasDigits = false
    var hasPredicate = false
    var hasOther = false
    for (c <- token) {
      hasDigits = hasDigits || c.isDigit
      hasPredicate = hasPredicate || predicate(c)
      hasOther = hasOther || !c.isLetterOrDigit
    }
    !hasOther && hasDigits && hasPredicate
  }

  def isAllSymbols(token: String) = {
    !token.forall(c => c.isLetterOrDigit)
  }

  def isShort(token: String) = {
    token.length == 2 && token(0).isUpper && token(1) == '.'
  }

  def containsUpper(token: String) = token.exists(c => c.isUpper)

  def containsLower(token: String) = token.exists(c => c.isLower)

  def containsLetter(token: String) = token.exists(c => c.isLetter)

  def containsDigit(token: String) = token.exists(c => c.isDigit)

  def containsSymbol(token: String) = token.exists(c => c.isLetterOrDigit)

  def getSuffix(token: String, size: Int, default: String = "") = {
    if (token.length >= size)
      token.substring(token.length - size).toLowerCase
    else
      default
  }

  def getPrefix(token: String, size: Int, default: String = "") = {
    if (token.length >= size)
      token.substring(0, size).toLowerCase
    else
      default
  }

  def fillFeatures(token: String): mutable.Map[String, String] = {
    val f = mutable.Map[String, String]()
    f("w") = token
    f("wl") = token.toLowerCase

    f("s") = getShape(token)
    f("h") = shrink(f("s"))
    f("t") = getType(token).toString

    f("p1") = getPrefix(token, 1)
    f("p2") = getPrefix(token, 2)
    f("p3") = getPrefix(token, 3)
    f("p4") = getPrefix(token, 4)

    f("s1") = getSuffix(token, 1)
    f("s2") = getSuffix(token, 2)
    f("s3") = getSuffix(token, 3)
    f("s4") = getSuffix(token, 4)


    f("dl") = isDigitOrPredicate(token, c => c.isLetter).toString
    f("d-") = isDigitOrPredicate(token, c => c == '-').toString
    f("d/") = isDigitOrPredicate(token, c => c == '/').toString
    f("d,") = isDigitOrPredicate(token, c => c == ',').toString
    f("d.") = isDigitOrPredicate(token, c => c == '.').toString

    f("u.") = isShort(token).toString
    f("iu") = (token.nonEmpty && token(0).isUpper).toString

    f("cu") = containsUpper(token).toString
    f("cl") = containsLower(token).toString
    f("ca") = containsLetter(token).toString
    f("cd") = containsDigit(token).toString
    f("cs") = containsSymbol(token).toString

    f
  }

  val pairs = Array("w", "pos", "h", "t")
  val window = 2

  def isInRange(idx: Int, size: Int) = idx >= 0 && idx < size

  def getName(source: String, idx: Int): String = {
    source + "~" + idx
  }

  def getName(source: String, idx1: Int, idx2: Int): String = {
    getName(source, idx1) + "|" + getName(source, idx2)
  }

  def generate(taggedSentence: TaggedSentence,
               wordpieceEmbeddingsSentence: WordpieceEmbeddingsSentence): TextSentenceAttrs = {

    val wordFeatures = taggedSentence.words
      .zip(taggedSentence.tags)
      .map{case (word, tag) =>
        val f = fillFeatures(word)
        f("pos") = tag
        f
      }

    val words = wordFeatures.length

    var wordsList = taggedSentence.words.toList
    val embeddings = wordpieceEmbeddingsSentence.tokens
      .filter(t => t.isWordStart)
      .map(t => t.embeddings)

    assert(embeddings.length == wordsList.length,
      "Mismatched embedding tokens and sentence tokens. Make sure you are properly " +
        "linking tokens and embeddings to the same inputCol DOCUMENT annotator")

    val attrs = (0 until words).map { i =>
      val pairAttrs = (-window until window)
        .filter(j => isInRange(i + j, words) && isInRange(i + j + 1, words))
        .flatMap(j =>
          pairs.map{name =>
            val feature = getName(name, j, j + 1)
            val value1 = wordFeatures(i + j).getOrElse(name, "")
            val value2 = wordFeatures(i + j + 1).getOrElse(name, "")
            (feature, value1 + "|" + value2)
          }
        ).toArray

      val unoAttrs = (-window to window)
        .filter(j => isInRange(i + j, words))
        .flatMap{j =>
          wordFeatures(i + j).map{case(name, value) =>
            (getName(name, j), value)
          }
        }.toArray

      val dictAttrs = dictFeatures.get(wordsList).map((getName("dt", i), _))
      wordsList = wordsList.tail

      val addition =
        if (i == 0) Array(("_BOS_", ""))
        else if (i == words - 1) Array(("_EOS_", ""))
        else Array.empty[(String, String)]

      val binAttrs = pairAttrs ++ unoAttrs ++ dictAttrs ++ addition

      val numAttrs = embeddings(i)

      WordAttrs(binAttrs, numAttrs)
    }

    TextSentenceAttrs(attrs)
  }

  def generateDataset(sentences:
                      TraversableOnce[(TextSentenceLabels, TaggedSentence, WordpieceEmbeddingsSentence)])
  : CrfDataset = {
    val textDataset = sentences
      .filter(p => p._2.words.length > 0)
      .map{case (labels, sentence, withEmbeddings) =>
        val textSentence = generate(sentence, withEmbeddings)
        (labels, textSentence)
      }

    DatasetReader.encodeDataset(textDataset)
  }

  def generate(sentence: TaggedSentence,
               withEmbeddings: WordpieceEmbeddingsSentence,
               metadata: DatasetMetadata): Instance = {
    val attrSentence = generate(sentence, withEmbeddings)

    DatasetReader.encodeSentence(attrSentence, metadata)
  }
}
