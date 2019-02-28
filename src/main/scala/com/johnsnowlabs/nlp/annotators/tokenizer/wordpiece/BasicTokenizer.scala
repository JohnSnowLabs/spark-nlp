package com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece

import java.text.Normalizer
import com.johnsnowlabs.nlp.annotators.common.{IndexedToken, Sentence}
import scala.collection.mutable


private [wordpiece] class BasicTokenizer(lowerCase: Boolean = true) {

  def isWhitespace(char: Char): Boolean = {
    char == ' ' || char == '\t' || char == '\n' || char == '\r' || Character.isWhitespace(char)
  }

  def isControl(char: Char): Boolean = {
    if (char == '\t' || char == '\n' || char == '\r')
      return false

    Character.isISOControl(char)
  }

  def isToFilter(char: Char): Boolean = {
    val cp = char.toInt
    cp == 0 || cp == 0xfffd || isControl(char)
  }

  def isPunctuation(char: Char): Boolean = {
    val cp = char.toInt
    if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
      (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126))
      return true

    val charCategory = Character.getName(char.toInt)
    charCategory.contains("PUNCTUATION")
  }

  def stripAccents(text: String): String = {
    Normalizer.normalize(text, Normalizer.Form.NFD)
      .replaceAll("\\p{InCombiningDiacriticalMarks}+", "")
  }

  def isChinese(char: Char): Boolean = {
    // This defines a "chinese character" as anything in the CJK Unicode block:
    //   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    //   https://github.com/google-research/bert/blob/master/tokenization.py

    val c = char.toInt

    (c >= 0x4E00 && c <= 0x9FFF) ||
      (c >= 0x3400 && c <= 0x4DBF) ||
      (c >= 0x20000 && c <= 0x2A6DF) ||
      (c >= 0x2A700 && c <= 0x2B73F) ||
      (c >= 0x2B740 && c <= 0x2B81F) ||
      (c >= 0x2B820 && c <= 0x2CEAF) ||
      (c >= 0xF900 && c <= 0xFAFF) ||
      (c >= 0x2F800 && c <= 0x2FA1F)
  }

  def normalize(text: String): String = {
    val result = stripAccents(text.trim())
      .filter(c => !isToFilter(c))
      .mkString("")

    if (lowerCase)
      result.toLowerCase
    else
      result
  }

  def tokenize(sentence: Sentence): Array[IndexedToken] = {

    val tokens = mutable.ArrayBuffer[IndexedToken]()
    val s = sentence.content

    def append(start: Int, end: Int): Unit = {
      assert(end > start)

      val text = s.substring(start, end)
      val normalized = normalize(text)

      if (!normalized.isEmpty) {
        val token = IndexedToken(normalized, start + sentence.start, end - 1 + sentence.start)
        tokens.append(token)
      }
    }

    var i = 0
    while (i < s.length) {
      // 1. Skip whitespaces
      while (isWhitespace(s(i)) && !isPunctuation(s(i)) && i < s.length)
        i = i + 1

      // 2. Find Next separator
      var end = i
      while (!isToFilter(s(end)) && !isPunctuation(s(end)) && !isChinese(s(end))
        && !isWhitespace(s(end)) && end < s.length)
        end += 1

      // 3. Detect what tokens to add
      if (end > i)
        append(i, end)

      if (isPunctuation(s(end)) || isChinese(s(end)))
        append(end, end + 1)

      i = end + 1
    }

    tokens.toArray
  }
}

