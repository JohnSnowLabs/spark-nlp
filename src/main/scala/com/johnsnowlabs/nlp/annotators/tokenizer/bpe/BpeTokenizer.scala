/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators.tokenizer.bpe

import com.johnsnowlabs.nlp.annotators.common.{IndexedToken, Sentence, TokenPiece}
import org.apache.commons.lang3.StringUtils

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/** A BPE Tokenizer based on GPT2's tokenization scheme. The tokenization can then be used for
 * models based on this scheme (e.g. GPT2, roBERTa, DeBERTa)
 *
 * TODO: truncation assumed?
 *
  * @param merges
  *   Map of tokens that are mergeable
  * @param vocab
  *   Map of tokens to encoded representation
  * @param specialTokens
  *   Collection of special tokens
 * @param padWithSequenceTokens
  *   Whether to pad the sentence with sentence tokens at the start and end
 * @param addPrefixSpaceToSentence
  *   Whether to add a space to the first word of a sentence
 * @param alwaysAddPrefix
 *    Whether to always prefix token ids with `prefixForPieceId`
  */
private[nlp] abstract class BpeTokenizer(
    val merges: Map[(String, String), Int],
    val vocab: Map[String, Int],
    val specialTokens: SpecialTokens,
    val padWithSequenceTokens: Boolean,
    val addPrefixSpaceToSentence: Boolean,
    val alwaysAddPrefix: Boolean) {

  protected val bpeRanks: Map[(String, String), Int] = {
    merges
  }

  /** Rankings for the byte pairs. Derived from merges.txt */
  protected def getBpeRanking: ((String, String)) => Int =
    (bytePair: (String, String)) => bpeRanks.getOrElse(bytePair, Integer.MAX_VALUE)

  /** cache for already encoded tokens */
  protected val cache: mutable.Map[String, Array[String]] = mutable.Map()

  /** Create a sequence of byte-pairs of the word */
  protected def getBytePairs(word: Array[String]): Array[(String, String)] = {
    val createPairs = (i: Int) => (word(i), word(i + 1))
    (0 until (word.length - 1)).map(createPairs).toArray
  }

  // Can be overridden in inherited class
  protected val prefixForPieceId: Option[String] = None
  protected val suffixForPieceId: Option[String] = None

  protected def performMerges(
      wordChars: Array[String],
      charPairs: Array[(String, String)]): Array[String] = {
    var word = wordChars
    var pairs = charPairs
    // get highest priority byte-pair first
    var bytePair: (String, String) =
      pairs.sortWith(getBpeRanking(_) < getBpeRanking(_))(0)
    var done = false
    // while we still have byte-pairs from our vocabulary
    while (bpeRanks.contains(bytePair) && !done) {
      val (first, second) = bytePair
      val newWord: ListBuffer[String] = ListBuffer()
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
            (word(i) == first) && (i < word.length - 1) && word(i + 1) == second
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
        bytePair = pairs.sortWith(getBpeRanking(_) < getBpeRanking(_))(0)
      }
    }
    word
  }

  protected def getTokenPieces(indToken: IndexedToken, word: Array[String]): Array[TokenPiece] = {
    var currentIndex = indToken.begin
    val wordIndexes = word.map((subWord: String) => {
      val startIndex = currentIndex
      currentIndex = startIndex + subWord.length
      (startIndex, startIndex + subWord.length - 1)
    })
    val result = word
      .zip(wordIndexes)
      .map { case (subWord: String, indexes: (Int, Int)) =>
        val isWordStart = indToken.begin == indexes._1
        val isDocumentStart = indToken.begin == 0
        var processedSubWord = subWord
        processedSubWord = if (isDocumentStart && !addPrefixSpaceToSentence) {
          processedSubWord
        } else
          prefixForPieceId match {
            case Some(prepend) if alwaysAddPrefix =>
              if (isWordStart && subWord.indexOf(prepend) < 0) prepend + processedSubWord
              else processedSubWord
            case _ => processedSubWord
          }
        processedSubWord = suffixForPieceId match {
          case None => processedSubWord
          case Some(append) =>
            val isWordEnd = indToken.end == indexes._2
            if (isWordEnd && subWord.indexOf(append) < 0) processedSubWord + append
            else processedSubWord
        }
        // Set unknown id if not found
        val subWordId: Int = vocab.getOrElse(processedSubWord, specialTokens.unk.id)

        TokenPiece(subWord, indToken.token.trim(), subWordId, isWordStart, indexes._1, indexes._2)

      }
    result
  }

  /** Do the BPE algorithm. Goal is to find the token as the largest words in the known
    * vocabulary. If not possible, the word is split into smaller subwords, until they are known.
    *
    * @return
    *   Array of TokenPieces, corresponding to encoded token
    */
  protected def bpe(indToken: IndexedToken): Array[TokenPiece] = {
    var processedToken = ""
    try {
      processedToken = preProcessTokenForBpe(indToken.token)
      // TODO: Caching
      var word: Array[String] = Array[String]()
      // split the word into characters, to be combined into subwords
      word = processedToken.map(_.toString).toArray
      val pairs: Array[(String, String)] = getBytePairs(word)

      if (pairs.isEmpty)
        word = Array(processedToken)
      else
        word = performMerges(word, pairs)

      getTokenPieces(indToken, word)
    } catch {
      case _: java.util.NoSuchElementException =>
        Array(
          TokenPiece(
            indToken.token,
            indToken.token,
            specialTokens.unk.id,
            isWordStart = true,
            indToken.begin,
            indToken.end))
    }
  }

  /** Split the the individual sub texts on special tokens, e.g. masking etc. */
  protected def splitOnSpecialToken(
      specialToken: SpecialToken,
      text: String): ListBuffer[String] = {
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

    val result: ListBuffer[String] = ListBuffer()
    val tok = specialToken.content

    val splitText = StringUtils.splitByWholeSeparator(text, tok)
    var fullWord = ""

    for ((subText, i) <- splitText.zipWithIndex) {
      var done = false
      // Try to avoid splitting on token
      if (specialToken.singleWord) {
        if ((i < (splitText.length - 1)) && !isEndOfWord(subText) && !isStartOfWord(
            splitText(i + 1))) fullWord += subText + tok
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

  /** Needs to be implemented */
  protected def tokenizeSubText(text: String, indexOffset: Int): Array[IndexedToken]

  /** Special tokens of the model for processing */
  val sentencePadding: (String, String) =
    (specialTokens.sentenceStart.content, specialTokens.sentenceEnd.content)

  /** Tokenize considering special tokens and split algorithm */
  def tokenize(sentence: Sentence): Array[IndexedToken] = {
    var text = sentence.content
    if (text.trim.isEmpty) Array[IndexedToken]()
    else {
      val splitTexts: ListBuffer[String] = ListBuffer()
      var textList: ListBuffer[String] = ListBuffer(text)

      for (transformations <- specialTokens.allTokens) {
        splitTexts.clear()
        for (subText <- textList) {
          if (!specialTokens.contains(subText))
            splitTexts ++= splitOnSpecialToken(transformations, subText)
          else
            splitTexts += subText
        }
        textList = splitTexts.clone()
      }

      if (padWithSequenceTokens) {
        text = sentencePadding._1 + text + sentencePadding._2
        splitTexts.prepend(sentencePadding._1)
        splitTexts.append(sentencePadding._2)
      }

      var currentIndex = 0
      val result = mutable.ArrayBuffer[IndexedToken]()
      for (subText <- splitTexts) {
        val subTextIndex = sentence.start + text.indexOf(subText, currentIndex)
        if (!specialTokens.contains(subText)) {
          val splitSubText: Array[IndexedToken] = tokenizeSubText(subText, subTextIndex)
          result.append(splitSubText: _*)
        } else // subtext is just the special token
          result.append(
            IndexedToken(subText, begin = subTextIndex, end = subTextIndex + subText.length - 1))
        currentIndex = subTextIndex + subText.length
      }
      result.toArray
    }
  }

  protected def preProcessTokenForBpe(token: String): String = token

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
          indToken.end))
  }

  def encode(indTokens: Array[IndexedToken]): Array[TokenPiece] = indTokens.flatMap(encode(_))
}

object BpeTokenizer {
  def forModel(
      modelType: String,
      merges: Map[(String, String), Int],
      vocab: Map[String, Int],
      padWithSequenceTokens: Boolean = false,
      addPrefixSpaceToSentence: Boolean = false,
      specialTokens: Option[SpecialTokens] = None,
      alwaysAddPrefix: Boolean = true): BpeTokenizer = {

    def modelSpecialTokens() = specialTokens match {
      case Some(specialTok) => specialTok
      case None => SpecialTokens.getSpecialTokensForModel(modelType, vocab)
    }

    val tokenizer = modelType match {
      case "roberta" =>
        new RobertaTokenizer(
          merges,
          vocab,
          modelSpecialTokens(),
          padWithSequenceTokens,
          addPrefixSpaceToSentence = addPrefixSpaceToSentence,
          alwaysAddPrefix = alwaysAddPrefix)
      case "xlm" =>
        new XlmTokenizer(merges, vocab, modelSpecialTokens(), padWithSequenceTokens)
      case "gpt2" =>
        new Gpt2Tokenizer(
          merges,
          vocab,
          modelSpecialTokens(),
          padWithSequenceTokens,
          addPrefixSpaceToSentence = addPrefixSpaceToSentence,
          alwaysAddPrefix = alwaysAddPrefix)
      case "bart" =>
        new BartTokenizer(
          merges,
          vocab,
          modelSpecialTokens(),
          padWithSequenceTokens,
          addPrefixSpaceToSentence = addPrefixSpaceToSentence)
      case _ =>
        throw new IllegalArgumentException("Model type \"" + modelType + "\" not supported yet.")
    }

    tokenizer
  }
}
