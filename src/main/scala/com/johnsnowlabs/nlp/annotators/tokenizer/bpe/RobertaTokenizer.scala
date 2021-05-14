package com.johnsnowlabs.nlp.annotators.tokenizer.bpe

import com.johnsnowlabs.nlp.annotators.common.{IndexedToken, Sentence, TokenPiece}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.util.matching.Regex

class RobertaTokenizer(
                        merges: Array[String],
                        vocab: Map[String, Int],
                        specialTokens: SpecialTokens,
                        padWithSentenceTokens: Boolean = false,
                      ) extends BpeTokenizer(merges, vocab, specialTokens) {

  /**
    * Mapping for bytes to a different set of unicode characters (especially white spaces).
    * This improved model performance for gpt-2. TODO: Only for gpt-2 type bpe
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
    * Special tokens of the model for processing
    */
  //  override val specialTokens: SpecialTokens = {
  //    val bpeSpecialTokens = new BpeSpecialTokens("roberta")
  //    bpeSpecialTokens.getSpecialTokens
  //  }
  val sentencePadding: (String, String) = (specialTokens.sentenceStart.content, specialTokens.sentenceEnd.content)

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
  override def tokenize(
                         sentence: Sentence
                       ): Array[IndexedToken] = {
    var text = sentence.content
    if (text.trim.isEmpty) Array[IndexedToken]()
    else {
      var splitTexts: ListBuffer[String] = ListBuffer()
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
      if (padWithSentenceTokens) {
        text = sentencePadding._1 + text + sentencePadding._2
        splitTexts.prepend(sentencePadding._1)
        splitTexts.append(sentencePadding._2)
      }
      var currentIndex = 0
      val result = mutable.ArrayBuffer[IndexedToken]()
      for (subText <- splitTexts) {
        val subTextIndex = sentence.start + text.indexOf(subText, currentIndex)
        if (!specialTokens.contains(subText)) {
          val splitSubText = splitOnPattern(subText, sentence.start + subTextIndex)
          result.append(splitSubText: _*)
        } else // subtext is just the special token
          result.append(
            IndexedToken(
              subText,
              begin = subTextIndex,
              end = subTextIndex + subText.length
            )
          )
        currentIndex = subTextIndex + subText.length
      }
      result.toArray
    }
  }

  override def encode(indToken: IndexedToken): Array[TokenPiece] = {
    if (!specialTokens.contains(indToken.token))
      bpe(indToken, encodeByte)
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
}
