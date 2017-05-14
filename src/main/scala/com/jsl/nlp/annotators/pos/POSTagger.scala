package com.jsl.nlp.annotators.pos

import com.jsl.nlp.annotators.Normalizer
import com.jsl.nlp.{Annotation, Annotator, Document}

/**
  * Created by Saif Addin on 5/13/2017.
  */
class POSTagger(taggingApproach: POSApproach) extends Annotator {

  private case class ToBeTagged(token: String, start: Int, end: Int)

  override val aType: String = POSTagger.aType

  override val requiredAnnotationTypes: Seq[String] = Seq(Normalizer.aType)

  override def annotate(document: Document, annotations: Seq[Annotation]): Seq[Annotation] = {
    val tokens: Array[ToBeTagged] = annotations.collect {
      case token: Annotation if token.aType == Normalizer.aType =>
        ToBeTagged(
          token.metadata.getOrElse(
            Normalizer.aType,
            throw new IllegalArgumentException(
              s"Annotation of type ${Normalizer.aType} does not provide proper token in metadata"
            )
          ),
          token.begin,
          token.end
        )
    }.toArray
    taggingApproach.tag(tokens.map(_.token)).zip(tokens).map{case (tagged, token) => {
      Annotation(
        POSTagger.aType,
        token.start,
        token.end,
        Map("word" -> tagged.word, "tag" -> tagged.tag)
      )
    }}
  }

}
object POSTagger {
  val aType = "pos"
}