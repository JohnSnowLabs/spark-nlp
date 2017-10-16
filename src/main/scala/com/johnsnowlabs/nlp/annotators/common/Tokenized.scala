package com.johnsnowlabs.nlp.annotators.common

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}

object Tokenized extends Annotated[TokenizedSentence] {

  override def annotatorType = AnnotatorType.TOKEN

  override def unpack(annotations: Seq[Annotation]): Seq[TokenizedSentence] = {
    val tokens = annotations
      .filter(_.annotatorType == annotatorType)
      .toArray
      .sortBy(a => a.begin)

    val tokenBegin = tokens.map(t => t.begin)
    val tokenEnd = tokens.map(t => t.end)

    def find(begin: Int, end: Int): Array[IndexedToken] = {
      import scala.collection.Searching._
      val beginIdx = tokenBegin.search(begin).insertionPoint
      val endIdx = tokenEnd.search(end + 1).insertionPoint

      val result = Array.fill[IndexedToken](endIdx - beginIdx)(null)
      for (i <- beginIdx until endIdx) {
        val token = tokens(i)
        result(i - beginIdx) = IndexedToken(token.metadata(annotatorType), token.begin, token.end)
      }

      result
    }

    SentenceSplit.unpack(annotations)
      .map(sentence => TokenizedSentence(find(sentence.begin, sentence.end)))
  }

  override def pack(sentences: Seq[TokenizedSentence]): Seq[Annotation] = {
    var sentenceIndex = 0

    sentences.flatMap{sentence =>
      sentenceIndex += 1
        sentence.indexedTokens.map{token =>
        Annotation(annotatorType, token.begin, token.end,
          Map(annotatorType -> token.token, "sentence" -> sentenceIndex.toString))
    }}
  }
}
