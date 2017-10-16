package com.johnsnowlabs.nlp.annotators.parser.dep

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition._
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
            (a1.annotatorType == POS || a1.annotatorType == TOKEN) &&
              a.metadata(Annotation.BEGIN).toInt <= a1.metadata(Annotation.BEGIN).toInt &&
              a.metadata(Annotation.END).toInt >= a1.metadata(Annotation.END).toInt
          }.groupBy( _.annotatorType )
        val tokens = tokensAndPosTags(TOKEN).sortBy { _.metadata(Annotation.BEGIN).toInt }
        val posTags = tokensAndPosTags(POS).sortBy { _.metadata(Annotation.BEGIN).toInt }
        val dependencies = model.parse(tokens, posTags)
        tokens
          .zip(dependencies)
          .map { case (token, index) => Annotation(
            DEPENDENCY,
            index.toString,
            token.metadata) }
      }
  }
}

object DependencyParserModel extends DefaultParamsReadable[DependencyParserModel]
