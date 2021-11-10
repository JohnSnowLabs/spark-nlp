package com.johnsnowlabs.nlp.annotators.multipleannotations

import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT}
import com.johnsnowlabs.nlp._
import org.apache.spark.ml.util.Identifiable


class MultiColumnsModel(override val uid: String) extends AnnotatorModel[MultiColumnsModel]
  with HasMultipleInputAnnotationCols
  with HasSimpleAnnotate[MultiColumnsModel]{

  def this() = this(Identifiable.randomUID("MERGE"))


  /**
   * Input annotator types: CHUNK
   *
   * @group anno
   */
  override val outputAnnotatorType: AnnotatorType = DOCUMENT


  /**
   * Multiple columns
   *
   * @group anno
   */

  override val inputAnnotatorType: String = DOCUMENT

  /**
   * Merges columns of chunk Annotations while considering false positives and replacements.
   * @param annotations a Sequence of chunks to merge
   * @return a Sequence of Merged CHUNK Annotations
   */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations
  }


}


