package com.johnsnowlabs.nlp.annotators.multipleannotations

import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT}
import com.johnsnowlabs.nlp._
import org.apache.spark.ml.util.Identifiable


class MultiColumnsModel(override val uid: String) extends AnnotatorModel[MultiColumnsModel]
  with HasMultipleInputAnnotationCols
  with HasSimpleAnnotate[MultiColumnsModel]{

  def this() = this(Identifiable.randomUID("MERGE"))

  /**
    * Input annotator types: DOCUMEN
    *
    */
  override val outputAnnotatorType: AnnotatorType = DOCUMENT
  /**
    * Output annotator type:DOCUMENT
    *
    */
  override val inputAnnotatorType: AnnotatorType = DOCUMENT


  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations
  }


}


