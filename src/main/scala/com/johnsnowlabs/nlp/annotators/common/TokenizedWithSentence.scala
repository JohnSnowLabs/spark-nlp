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
      ).map(token => IndexedToken(token.result, token.begin, token.end, token.metadata.get("sentence")))
      sentenceTokens
    }).filter(_.nonEmpty).map(indexedTokens => TokenizedSentence(indexedTokens))

  }

  override def pack(sentences: Seq[TokenizedSentence]): Seq[Annotation] = {
    var sentenceIndex = 0
    // TODO: don't access global state from inside a map!! - use zipWithindex instead
    sentences.flatMap{sentence =>
      sentenceIndex += 1
        sentence.indexedTokens.map{token =>
        Annotation(annotatorType, token.begin, token.end, token.token,
          Map("sentence" -> sentenceIndex.toString))
    }}
  }
}