package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.annotators.common.{IndexedTaggedWord, TaggedSentence, TaggedWord}
import com.johnsnowlabs.nlp.annotators.ws.TagsType.{LEFT_BOUNDARY, MIDDLE, RIGHT_BOUNDARY, SINGLE_WORD}
import com.johnsnowlabs.nlp.annotators.ws.WordSegmenterModel
import org.scalatest.FlatSpec


class WordSegmenterModelTest extends FlatSpec {

  "A Word Segmenter Model that predicts all tags right" should "build a word segment" in {
    val taggedWords = Array(TaggedWord("有", LEFT_BOUNDARY), TaggedWord("限", MIDDLE),
      TaggedWord("公", MIDDLE), TaggedWord("司", RIGHT_BOUNDARY))
    val indexedTaggedWords = Array(IndexedTaggedWord("有", LEFT_BOUNDARY, 0, 0, None, Map("index" -> "0")),
      IndexedTaggedWord("限", MIDDLE, 1, 1, None, Map("index" -> "1")),
      IndexedTaggedWord("公", MIDDLE, 2, 2, None, Map("index" -> "2")),
      IndexedTaggedWord("司", RIGHT_BOUNDARY, 3, 3, None, Map("index" -> "3")))
    val taggedSentences = Array(TaggedSentence(taggedWords, indexedTaggedWords))
    val expectedWordSegments = Seq(
      Annotation(TOKEN, 0, 3, "有限公司", Map("sentence" -> "0"))
    )

    val actualWordSegments = new WordSegmenterModel().buildWordSegments(taggedSentences)

    assert(actualWordSegments == expectedWordSegments)
  }

  "A Word Segmenter Model that does not predict any tag right for all elements" should "build only single words" in {

    val taggedWords = Array(TaggedWord("永", MIDDLE), TaggedWord("和", MIDDLE), TaggedWord("服", MIDDLE))
    val indexedTaggedWords = Array(IndexedTaggedWord("永", MIDDLE, 0, 0, None, Map("index" -> "0")),
      IndexedTaggedWord("和", MIDDLE, 1, 1, None, Map("index" -> "1")),
      IndexedTaggedWord("服", MIDDLE, 2, 2, None, Map("index" -> "2")))
    val taggedSentences = Array(TaggedSentence(taggedWords, indexedTaggedWords))
    val expectedWordSegments = Seq(Annotation(TOKEN, 0, 0, "永", Map("sentence" -> "0")),
      Annotation(TOKEN, 1, 1, "和", Map("sentence" -> "0")),
      Annotation(TOKEN, 2, 2, "服", Map("sentence" -> "0")))

    val actualWordSegments = new WordSegmenterModel().buildWordSegments(taggedSentences)

    assert(actualWordSegments == expectedWordSegments)
  }

  "A Word Segmenter Model that predicts only one right word" should "build a word segment" in {

    val taggedWords = Array(TaggedWord("永", LEFT_BOUNDARY), TaggedWord("和", RIGHT_BOUNDARY),
      TaggedWord("服", MIDDLE), TaggedWord("服", LEFT_BOUNDARY))
    val indexedTaggedWords = Array(IndexedTaggedWord("永", LEFT_BOUNDARY, 0, 0, None, Map("index" -> "0")),
      IndexedTaggedWord("和", RIGHT_BOUNDARY, 1, 1, None, Map("index" -> "1")),
      IndexedTaggedWord("服", MIDDLE, 2, 2, None, Map("index" -> "2")),
      IndexedTaggedWord("服", LEFT_BOUNDARY, 3, 3, None, Map("index" -> "3")))
    val taggedSentences = Array(TaggedSentence(taggedWords, indexedTaggedWords))
    val expectedWordSegments = Seq(
      Annotation(TOKEN, 0, 1, "永和", Map("sentence" -> "0")),
      Annotation(TOKEN, 2, 2, "服", Map("sentence" -> "0")),
      Annotation(TOKEN, 3, 3, "服", Map("sentence" -> "0"))
    )

    val actualWordSegments = new WordSegmenterModel().buildWordSegments(taggedSentences)

    assert(actualWordSegments == expectedWordSegments)
  }

  "A Word Segmenter Model that predicts right words in between a sentence" should "build a word segment" in {

    val taggedWords = Array(TaggedWord("永", LEFT_BOUNDARY), TaggedWord("和", RIGHT_BOUNDARY), TaggedWord("服", MIDDLE),
      TaggedWord("有", LEFT_BOUNDARY), TaggedWord("限", MIDDLE), TaggedWord("公", RIGHT_BOUNDARY),
      TaggedWord("服", LEFT_BOUNDARY), TaggedWord("奥", LEFT_BOUNDARY), TaggedWord("田", RIGHT_BOUNDARY))
    val indexedTaggedWords = Array(
      IndexedTaggedWord("永", LEFT_BOUNDARY, 0, 0, None, Map("index" -> "0")),
      IndexedTaggedWord("和", RIGHT_BOUNDARY, 1, 1, None, Map("index" -> "1")),
      IndexedTaggedWord("服", MIDDLE, 2, 2, None, Map("index" -> "2")),
      IndexedTaggedWord("有", LEFT_BOUNDARY, 3, 3, None, Map("index" -> "3")),
      IndexedTaggedWord("限", MIDDLE, 4, 4, None, Map("index" -> "4")),
      IndexedTaggedWord("公", RIGHT_BOUNDARY, 5, 5, None, Map("index" -> "5")),
      IndexedTaggedWord("服", SINGLE_WORD, 6, 6, None, Map("index" -> "6")),
      IndexedTaggedWord("奥", LEFT_BOUNDARY, 7, 7, None, Map("index" -> "7")),
      IndexedTaggedWord("田", RIGHT_BOUNDARY, 8, 8, None, Map("index" -> "8")))
    val taggedSentences = Array(TaggedSentence(taggedWords, indexedTaggedWords))
    val expectedWordSegments = Seq(
      Annotation(TOKEN, 0, 1, "永和", Map("sentence" -> "0")),
      Annotation(TOKEN, 2, 2, "服", Map("sentence" -> "0")),
      Annotation(TOKEN, 3, 5, "有限公", Map("sentence" -> "0")),
      Annotation(TOKEN, 6, 6, "服", Map("sentence" -> "0")),
      Annotation(TOKEN, 7, 8, "奥田", Map("sentence" -> "0"))
    )

    val actualWordSegments = new WordSegmenterModel().buildWordSegments(taggedSentences)

    assert(actualWordSegments == expectedWordSegments)
  }

  "A word segment that predicts wrong words at the beginning of a sentence" should "build a word segment" in {
    val taggedWords = Array(TaggedWord("永", LEFT_BOUNDARY), TaggedWord("和", MIDDLE), TaggedWord("服", MIDDLE),
      TaggedWord("有", LEFT_BOUNDARY), TaggedWord("限", MIDDLE), TaggedWord("公", RIGHT_BOUNDARY))
    val indexedTaggedWords = Array(
      IndexedTaggedWord("永", LEFT_BOUNDARY, 0, 0, None, Map("index" -> "0")),
      IndexedTaggedWord("和", MIDDLE, 1, 1, None, Map("index" -> "1")),
      IndexedTaggedWord("服", MIDDLE, 2, 2, None, Map("index" -> "2")),
      IndexedTaggedWord("有", LEFT_BOUNDARY, 3, 3, None, Map("index" -> "3")),
      IndexedTaggedWord("限", MIDDLE, 4, 4, None, Map("index" -> "4")),
      IndexedTaggedWord("公", RIGHT_BOUNDARY, 5, 5, None, Map("index" -> "5"))
    )
    val taggedSentences = Array(TaggedSentence(taggedWords, indexedTaggedWords))
    val expectedWordSegments = Seq(
      Annotation(TOKEN, 0, 0, "永", Map("sentence" -> "0")),
      Annotation(TOKEN, 1, 1, "和", Map("sentence" -> "0")),
      Annotation(TOKEN, 2, 2, "服", Map("sentence" -> "0")),
      Annotation(TOKEN, 3, 5, "有限公", Map("sentence" -> "0"))
    )

    val actualWordSegments = new WordSegmenterModel().buildWordSegments(taggedSentences)

    assert(actualWordSegments == expectedWordSegments)

  }

  "A word segment that predicts wrong words at the end of a sentence" should "build a word segment" in {
    val taggedWords = Array(TaggedWord("有", LEFT_BOUNDARY), TaggedWord("限", MIDDLE), TaggedWord("公", RIGHT_BOUNDARY),
      TaggedWord("永", LEFT_BOUNDARY), TaggedWord("和", MIDDLE), TaggedWord("服", MIDDLE))
    val indexedTaggedWords = Array(
      IndexedTaggedWord("有", LEFT_BOUNDARY, 0, 0, None, Map("index" -> "0")),
      IndexedTaggedWord("限", MIDDLE, 1, 1, None, Map("index" -> "1")),
      IndexedTaggedWord("公", RIGHT_BOUNDARY, 2, 2, None, Map("index" -> "2")),
      IndexedTaggedWord("永", LEFT_BOUNDARY, 3, 3, None, Map("index" -> "3")),
      IndexedTaggedWord("和", MIDDLE, 4, 4, None, Map("index" -> "4")),
      IndexedTaggedWord("服", MIDDLE, 5, 5, None, Map("index" -> "5"))
    )
    val taggedSentences = Array(TaggedSentence(taggedWords, indexedTaggedWords))
    val expectedWordSegments = Seq(
      Annotation(TOKEN, 0, 2, "有限公", Map("sentence" -> "0")),
      Annotation(TOKEN, 3, 3, "永", Map("sentence" -> "0")),
      Annotation(TOKEN, 4, 4, "和", Map("sentence" -> "0")),
      Annotation(TOKEN, 5, 5, "服", Map("sentence" -> "0"))
    )

    val actualWordSegments = new WordSegmenterModel().buildWordSegments(taggedSentences)

    assert(actualWordSegments == expectedWordSegments)

  }

  "A Word Segmenter Model that predicts wrong words in between a sentence" should "build a word segment" in {
    //永和 服 装饰 。品 有限公司
    val taggedWords = Array(TaggedWord("永", LEFT_BOUNDARY), TaggedWord("和", RIGHT_BOUNDARY),
      TaggedWord("服", MIDDLE),
      TaggedWord("装", LEFT_BOUNDARY), TaggedWord("饰", RIGHT_BOUNDARY),
      TaggedWord("。", MIDDLE),
      TaggedWord("品", LEFT_BOUNDARY),
      TaggedWord("有", LEFT_BOUNDARY), TaggedWord("限", MIDDLE), TaggedWord("公", MIDDLE), TaggedWord("司", RIGHT_BOUNDARY)
    )
    val indexedTaggedWords = Array(
      IndexedTaggedWord("永", LEFT_BOUNDARY, 0, 0, None, Map("index" -> "0")),
      IndexedTaggedWord("和", RIGHT_BOUNDARY, 1, 1, None, Map("index" -> "1")),
      IndexedTaggedWord("服", MIDDLE, 2, 2, None, Map("index" -> "2")),
      IndexedTaggedWord("装", LEFT_BOUNDARY, 3, 3, None, Map("index" -> "3")),
      IndexedTaggedWord("饰", RIGHT_BOUNDARY,4, 4, None, Map("index" -> "4")),
      IndexedTaggedWord("。", MIDDLE, 5, 5, None, Map("index" -> "5")),
      IndexedTaggedWord("品", LEFT_BOUNDARY, 6, 6, None, Map("index" -> "6")),
      IndexedTaggedWord("有", LEFT_BOUNDARY, 7, 7, None, Map("index"-> "7")),
      IndexedTaggedWord("限", MIDDLE, 8, 8, None, Map("index" -> "8")),
      IndexedTaggedWord("公", MIDDLE, 9, 9, None, Map("index" -> "9")),
      IndexedTaggedWord("司", RIGHT_BOUNDARY, 10,10, None, Map("index" -> "10"))
    )
    val taggedSentences = Array(TaggedSentence(taggedWords, indexedTaggedWords))
    val expectedWordSegments = Seq(
      Annotation(TOKEN, 0, 1, "永和", Map("sentence" -> "0")),
      Annotation(TOKEN, 2, 2, "服", Map("sentence" -> "0")),
      Annotation(TOKEN, 3, 4, "装饰", Map("sentence" -> "0")),
      Annotation(TOKEN, 5, 5, "。", Map("sentence" -> "0")),
      Annotation(TOKEN, 6, 6, "品", Map("sentence" -> "0")),
      Annotation(TOKEN, 7, 10, "有限公司", Map("sentence" -> "0"))
    )

    val actualWordSegments = new WordSegmenterModel().buildWordSegments(taggedSentences)

    assert(actualWordSegments == expectedWordSegments)

  }

}
