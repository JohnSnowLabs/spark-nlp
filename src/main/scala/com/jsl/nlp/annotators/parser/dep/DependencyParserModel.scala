package com.jsl.nlp.annotators.parser.dep

import com.jsl.nlp.{Annotation, AnnotatorModel}
import com.jsl.nlp.AnnotatorType._
import com.jsl.nlp.annotators.parser.dep.GreedyTransition._
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

class DependencyParserModel(override val uid: String) extends AnnotatorModel[DependencyParserModel] {
  def this() = this(Identifiable.randomUID(DEPENDENCY))

  override val annotatorType: String = DEPENDENCY

  override val requiredAnnotatorTypes =  Array[String](DOCUMENT, POS, TOKEN)

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val model = new GreedyTransitionApproach
    annotations
      .filter { _.annotatorType == DOCUMENT }
      .flatMap { a: Annotation =>
        val tokensAndPosTags: Map[String, Seq[Annotation]] = annotations
          .filter { a1 =>
            (a1.annotatorType == POS || a1.annotatorType == TOKEN) && a.begin <= a1.begin && a.end >= a1.end
          }.groupBy( _.annotatorType )
        val tokens = tokensAndPosTags(TOKEN).sortBy { _.begin }
        val posTags = tokensAndPosTags(POS).sortBy { _.begin }
        val dependencies = model.parse(tokens, posTags)
        tokens
          .zip(dependencies)
          .map { case (token, index) => Annotation(DEPENDENCY, token.begin, token.end, Map("head" -> index.toString )) }
      }
  }
}

object DependencyParserModel extends DefaultParamsReadable[DependencyParserModel]
