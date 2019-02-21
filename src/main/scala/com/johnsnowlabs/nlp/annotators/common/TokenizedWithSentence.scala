package com.johnsnowlabs.nlp.annotators.common

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}

object TokenizedWithSentence extends Annotated[TokenizedSentence] {

  override def annotatorType = AnnotatorType.TOKEN

  override def unpack(annotations: Seq[Annotation]): Seq[TokenizedSentence] = {
    val tokens = annotations
      .filter(_.annotatorType == annotatorType)
      .toArray

    SentenceSplit.unpack(annotations).map(sentence => {
      // TODO: don't traverse every token every time, only one pass should be enough
      // hint: first get the sentence limits, and then use them to do only one pass of 'tokens',
      // make this explode in the number of sentences(small), not the number of tokens(huge)
      val sentenceTokens = tokens.filter(token =>
        // TODO: replaced &, don't need evaluation of second condition when first is false
        token.begin >= sentence.start && token.end <= sentence.end
      ).map(token => IndexedToken(token.result, token.begin, token.end, token.metadata("sentence").toInt))
      (sentenceTokens, sentence.id)
    }).filter(_._1.nonEmpty).map(indexedTokens => TokenizedSentence(indexedTokens._1, indexedTokens._2))

  }

  override def pack(sentences: Seq[TokenizedSentence]): Seq[Annotation] = {
    sentences.flatMap{sentence =>
        sentence.indexedTokens.map{token =>
        Annotation(annotatorType, token.begin, token.end, token.token,
          Map("sentence" -> token.sentenceId.toString))
    }}
  }
}