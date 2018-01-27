package com.johnsnowlabs.nlp.annotators.common

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}

object Tokenized extends Annotated[TokenizedSentence] {

  override def annotatorType = AnnotatorType.TOKEN

  override def unpack(annotations: Seq[Annotation]): Seq[TokenizedSentence] = {
    val tokens = annotations
      .filter(_.annotatorType == annotatorType)
      .toArray

    SentenceSplit.unpack(annotations).map(sentence => {
      tokens.filter(token =>
        token.start >= sentence.start & token.end <= sentence.end
      ).map(token => IndexedToken(token.result, token.start, token.end))
    }).filter(_.nonEmpty).map(indexedTokens => TokenizedSentence(indexedTokens))

  }

  override def pack(sentences: Seq[TokenizedSentence]): Seq[Annotation] = {
    var sentenceIndex = 0

    sentences.flatMap{sentence =>
      sentenceIndex += 1
        sentence.indexedTokens.map{token =>
        Annotation(annotatorType, token.begin, token.end, token.token,
          Map("sentence" -> sentenceIndex.toString))
    }}
  }
}
