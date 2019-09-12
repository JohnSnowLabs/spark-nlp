package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

class Token2Chunk(override val uid: String) extends AnnotatorModel[Token2Chunk]{

  override val outputAnnotatorType: AnnotatorType = CHUNK

  override val inputAnnotatorTypes: Array[String] = Array(TOKEN)

  def this() = this(Identifiable.randomUID("TOKEN2CHUNK"))

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.map { token =>
      Annotation(
        CHUNK,
        token.begin,
        token.end,
        token.result,
        token.metadata
      )
    }
  }

}

object Token2Chunk extends DefaultParamsReadable[Token2Chunk]