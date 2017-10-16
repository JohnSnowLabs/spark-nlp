package com.johnsnowlabs.nlp

import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
  * Created by saif on 06/07/17.
  */

class TokenAssembler(override val uid: String) extends AnnotatorModel[TokenAssembler]{

  import com.johnsnowlabs.nlp.AnnotatorType._

  override val annotatorType: AnnotatorType = DOCUMENT

  override val requiredAnnotatorTypes: Array[String] = Array(TOKEN)

  def this() = this(Identifiable.randomUID("TOKEN_ASSEMBLER"))

  override protected def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.groupBy(token => token.result)
      .map{case (_, sentenceAnnotations) =>
          Annotation(
            DOCUMENT,
            sentenceAnnotations.map(_.result).mkString(" "),
            Map(
              Annotation.BEGIN -> sentenceAnnotations.minBy(_.metadata(Annotation.BEGIN)).metadata(Annotation.BEGIN),
              Annotation.END -> sentenceAnnotations.maxBy(_.metadata(Annotation.END)).metadata(Annotation.END)
            )
          )
      }.toSeq
  }

}
object TokenAssembler extends DefaultParamsReadable[TokenAssembler]