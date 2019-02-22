package com.johnsnowlabs.nlp.annotators.common

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}

object TokenizedWithSentence extends Annotated[TokenizedSentence] {

  override def annotatorType = AnnotatorType.TOKEN

  override def unpack(annotations: Seq[Annotation]): Seq[TokenizedSentence] = {
    val tokens = annotations
      .filter(_.annotatorType == annotatorType)
      .toArray

    SentenceSplit.unpack(annotations).map(sentence => {
      val sentenceTokens = tokens.filter(token =>
        token.begin >= sentence.start & token.end <= sentence.end
      ).map(token => IndexedToken(token.result, token.begin, token.end))
      sentenceTokens
    }).filter(_.nonEmpty).zipWithIndex.map{case (indexedTokens, index) => TokenizedSentence(indexedTokens, index)}

  }

  override def pack(sentences: Seq[TokenizedSentence]): Seq[Annotation] = {
    sentences.zipWithIndex.flatMap{case (sentence, sentenceIndex) =>
        sentence.indexedTokens.map{token =>
        Annotation(annotatorType, token.begin, token.end, token.token,
          Map("sentence" -> sentenceIndex.toString))
    }}
  }
}