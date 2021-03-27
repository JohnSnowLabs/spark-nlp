package com.johnsnowlabs.nlp.annotators.tokenizer.bpe

import com.johnsnowlabs.nlp.annotators.common.{IndexedToken, Sentence, TokenPiece}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.util.matching.Regex

/**
  * A BPE Tokenizer based on GPT2's tokenization scheme.
  * The tokenization can then be used for models based on this scheme (e.g. GPT2, roBERTa, DeBERTa)
  * TODO: truncation assumed?
  */
private[nlp] class BpeTokenizer(
    merges: Array[String],
    vocab: Map[String, Int],
    modelType: String = "roberta",
    padWithSentenceTokens: Boolean = false
) {

  val bpeRanks: Map[(String, String), Int] = {
    val bytePairs: Array[(String, String)] =
      merges.map(_.split(" ")).map { case Array(c1, c2) => (c1, c2) }
    bytePairs.zipWithIndex.toMap
  }

  /**
    * Rankings for the byte pairs. Derived from merges.txt
    */
  def getBpeRanking: ((String, String)) => Int =
    (bytePair: (String, String)) => bpeRanks.getOrElse(bytePair, Integer.MAX_VALUE)

  /**
    * cache for already encoded tokens
    */
  private val cache: mutable.Map[String, Array[TokenPiece]] = mutable.Map()

  /**
    * Mapping for bytes to a different set of unicode characters (especially white spaces).
    * This improved model performance for gpt-2.
    */
  private val bytesToUnicodeMapping: Map[Int, String] = {
    val bytes: ListBuffer[Int] = ListBuffer.range(
      '!',
      '~' + 1
    ) ++ ListBuffer.range('¡', '¬' + 1) ++ ListBuffer.range('®', 'ÿ' + 1)
    val characters: ListBuffer[Int] = bytes.clone
    var n = 0
    for (b <- 0 to 256) {
      if (!bytes.contains(b)) {
        bytes += b
        characters += (256 + n)
        n += 1
      }
    }
    (bytes zip characters.map(_.toChar.toString)).toMap
  }
  private val encodeByte =
    (tok: String) => tok.foldLeft("")(_ + bytesToUnicodeMapping(_))

  /**
    * Create a sequence of byte-pairs of the word
    */
  private def getBytePairs(word: Array[String]): Set[(String, String)] = {
    val createPairs = (i: Int) => (word(i), word(i + 1))
    (0 until (word.length - 1)).map(createPairs).toSet
  }

  /**
    * Do the BPE algorithm. Goal is to find the token as the largest words in the known vocabulary.
    * If not possible, the word is split into smaller subwords, until they are known.
    * @return Array of TokenPieces, corresponding to encoded token
    */
  private def bpe(indToken: IndexedToken): Array[TokenPiece] = {
    val encodedToken = encodeByte(indToken.token)
    if (cache.contains(encodedToken))
      cache(encodedToken)
    else {
      // split the word into characters, to be combined into subwords
      var word: Array[String] = encodedToken.map(_.toString).toArray
      var pairs: Set[(String, String)] = getBytePairs(word)
      if (pairs.isEmpty) word = Array(encodedToken) // TODO: check if correct
      else {
        // get highest priority byte-pair first
        var bytePair: (String, String) =
          pairs.toArray.sortWith(getBpeRanking(_) < getBpeRanking(_))(0)
        var done = false
        // while we still have byte-pairs from our vocabulary
        while (bpeRanks.contains(bytePair) && !done) {
          val (first, second) = bytePair
          var newWord: ListBuffer[String] = ListBuffer()
          var i = 0
          var j = 0
          // keep combining characters with the current byte-pair
          while ((i < word.length) && (j != -1)) {
            j = word.indexOf(first, i)
            if (j == -1) newWord ++= word.drop(i)
            else {
              newWord ++= word.slice(i, j)
              i = j
              val bpIsAtIndex =
                (word(i) == first) && (i < word.length - 1) && word(
                  i + 1
                ) == second
              if (bpIsAtIndex) {
                newWord += (first + second)
                i += 2
              } else {
                newWord += word(i)
                i += 1
              }
            }
          }
          word = newWord.toArray
          // if we were able to create a whole word that was in the vocabulary, we're done
          if (word.length == 1) {
            done = true
          } else {
            // do it again with the next byte-pair
            pairs = getBytePairs(word)
            bytePair = pairs.toArray.sortWith(getBpeRanking(_) < getBpeRanking(_))(0)
          }
        }
      }
      val indexOffset = indToken.begin
      val wordIndexes = word.map((subWord: String) => {
        val startIndex = encodedToken.indexOf(subWord) + indexOffset
        (startIndex, startIndex + subWord.length) // TODO
      })
      val result = word
        .zip(wordIndexes)
        .map {
          case (subWord: String, indexes: (Int, Int)) =>
            val isWordStart = encodedToken.head == subWord.head
            require(vocab.contains(subWord), "token \"" + subWord + "\" not found in vocabulary")
            TokenPiece(subWord, encodedToken, vocab(subWord), isWordStart, indexes._1, indexes._2)
        }
      cache += (encodedToken -> result)
      result
    }
  }

  /**
    * Split the the individual sub texts on special tokens, e.g. masking etc.
    */
  private def splitOnSpecialToken(
      specialToken: TokenTransformations,
      text: String
  ): ListBuffer[String] = {
    val isControl = (c: Char) => {
      if (c == '\t' || c == '\n' || c == '\r') false // count as whitespace
      else c.isControl
    }
    val isPunctuation =
      (c: Char) => raw"""[^[:alnum:]]""".r.findFirstIn(c.toString).isDefined
    val isWordBorder =
      (c: Char) => isControl(c) || isPunctuation(c) || c.isWhitespace

    val isEndOfWord = (text: String) => isWordBorder(text.last)
    val isStartOfWord = (text: String) => isWordBorder(text.head)

    var result: ListBuffer[String] = ListBuffer()
    val tok = specialToken.content
    val splitText = text.split(tok)
    var fullWord = ""
    //    val boolProperty = (property: Map[String, Any], key: String) => property(key).asInstanceOf[Boolean]

    for ((subText, i) <- splitText.zipWithIndex) {
      var done = false
      // Try to avoid splitting on token
      if (specialToken.singleWord) {
        if (
          (i < (splitText.length - 1)) && !isEndOfWord(
            subText
          ) && !isStartOfWord(splitText(i + 1))
        ) fullWord += subText + tok
        else if (fullWord.nonEmpty) {
          fullWord += subText
          result += fullWord
          fullWord = ""
          done = true
        }
      }
      if (!done) {
        // A bit counter-intuitive but we strip the left of the string
        // since rstrip means the special token is eating all white spaces on its right
        var subTextProcessed: String = subText
        if (specialToken.rstrip && i > 0)
          subTextProcessed = subText.stripPrefix(" ")
        if (specialToken.lstrip && i < (splitText.length - 1))
          subTextProcessed = subText.stripSuffix(" ")
        if (i == 0 && subTextProcessed.isEmpty)
          result += tok
        else if (i == (splitText.length - 1)) {
          if (subTextProcessed.nonEmpty) result += subTextProcessed
        } else {
          if (subTextProcessed.nonEmpty) result += subTextProcessed
          result += tok
        }
      }
    }
    result
  }

  /**
    * Special tokens of the model for processing
    */
  val (specialTokens: Map[String, TokenTransformations], sentencePadding: (String, String)) = {
    val bpeSpecialTokens = new BpeSpecialTokens(modelType)
    (bpeSpecialTokens.getSpecialTokens, bpeSpecialTokens.getSentencePadding)
  }

  /**
    * split pattern based on gpt2's bpe tokenizer
    */
  private def splitOnPattern(text: String, indexOffset: Int): Array[IndexedToken] = {
    val splitPattern: Regex = raw"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+".r
    splitPattern
      .findAllMatchIn(text)
      .map(tok => IndexedToken(tok.matched, tok.start + indexOffset, tok.end + indexOffset)) // TODO Expected -1?
      .toArray
  }

  /**
    * Tokenize considering special tokens and split pattern
    */
  def tokenize(
      sentence: Sentence
  ): Array[IndexedToken] = {
    var text = sentence.content
    if (text.trim.isEmpty) Array[IndexedToken]()
    else {
      var splitTexts: ListBuffer[String] = ListBuffer()
      var textList: ListBuffer[String] = ListBuffer(text)

      for ((_, transformations) <- specialTokens) {
        splitTexts.clear()
        for (subText <- textList) {
          if (!specialTokens.contains(subText))
            splitTexts ++= splitOnSpecialToken(transformations, subText)
          else
            splitTexts += subText
        }
        textList = splitTexts.clone()
      }
      if (padWithSentenceTokens) {
        text = sentencePadding._1 + text + sentencePadding._2
        splitTexts.prepend(sentencePadding._1)
        splitTexts.append(sentencePadding._2)
      }
      val result = mutable.ArrayBuffer[IndexedToken]()
      for (subText <- splitTexts) {
        val subTextIndex = text.indexOf(subText)
        if (!specialTokens.contains(subText)) {
          val splitSubText = splitOnPattern(subText, sentence.start + subTextIndex)
          result.append(splitSubText: _*)
        } else // subtext is just the special token
          result.append(
            IndexedToken(
              subText,
              begin = sentence.start + subTextIndex,
              end = sentence.start + subTextIndex + subText.length // TODO
            )
          )
      }
      result.toArray
    }
  }

  def encode(indToken: IndexedToken): Array[TokenPiece] = {
    if (!specialTokens.contains(indToken.token))
      bpe(indToken)
    else
      Array(
        TokenPiece(
          indToken.token,
          indToken.token,
          vocab(indToken.token),
          isWordStart = true,
          indToken.begin,
          indToken.end
        )
      )
  }
  def encode(indTokens: Array[IndexedToken]): Array[TokenPiece] = indTokens.flatMap(encode(_))
}
