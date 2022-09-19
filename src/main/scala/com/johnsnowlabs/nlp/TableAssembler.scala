package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, TABLE}
import org.apache.spark.ml.util.Identifiable

class TableAssembler (override val uid: String)
  extends AnnotatorModel[TokenAssembler]
    with HasSimpleAnnotate[TokenAssembler] {

  def this() = this(Identifiable.randomUID("TABLE_ASSEMBLER"))


  /** Output annotator types: DOCUMENT
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = TABLE

  /** Input annotator types: DOCUMENT, TOKEN
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT)

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    annotations
      .map(annotation => new Annotation(
        annotatorType = TABLE,
        begin = annotation.begin,
        end = annotation.end,
        result = annotation.result,
        metadata = annotation.metadata))
  }
}
