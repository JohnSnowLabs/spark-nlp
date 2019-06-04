package com.johnsnowlabs.nlp.annotators.common

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}


object TokenizedWithSentence extends Annotated[TokenizedSentence] {

  override def annotatorType: String = AnnotatorType.TOKEN

  override def unpack(annotations: Seq[Annotation]): Seq[TokenizedSentence] = {
    val tokens = annotations
      .filter(_.annotatorType == annotatorType)
      .toArray

    val sentences = SentenceSplit.unpack(annotations)

    /** // Evaluate whether to enable this validation to check proper usage of DOCUMENT and SENTENCE within entire pipelines
    require(tokens.map(_.metadata.getOrElse("sentence", "0").toInt).distinct.length == sentences.length,
      "Inconsistencies found in pipeline. Tokens in sentences does not match with sentence count")
      */

    sentences.map(sentence => {
      val sentenceTokens = tokens.filter(token =>
        token.begin >= sentence.start & token.end <= sentence.end
      ).map(token => IndexedToken(token.result, token.begin, token.end))
      sentenceTokens
    }).filter(_.nonEmpty).zipWithIndex.map{case (indexedTokens, index) => TokenizedSentence(indexedTokens, index)}

  }

  override def pack(sentences: Seq[TokenizedSentence]): Seq[Annotation] = {
    sentences.zipWithIndex.flatMap{case (sentence, index) =>
        sentence.indexedTokens.map{token =>
        Annotation(annotatorType, token.begin, token.end, token.token,
          Map("sentence" -> index.toString))
    }}
  }
}
