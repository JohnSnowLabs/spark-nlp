package com.johnsnowlabs.nlp.annotators.spell.ocr

import com.johnsnowlabs.ml.tensorflow.TensorflowWrapper
import com.johnsnowlabs.nlp.{AnnotatorApproach, AnnotatorType}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset
import org.tensorflow.{Graph, Session}

class OcrSpellCheckApproach(override val uid: String) extends AnnotatorApproach[OcrSpellCheckModel]{
  override val description: String = "Ocr specific Spell Checking"

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): OcrSpellCheckModel = {

    val graph = new Graph()
    val config = Array[Byte](56, 1)
    val session = new Session(graph, config)
    val tf = new TensorflowWrapper(session, graph)

    // TODO: replace hard coded stuff for development only
    new OcrSpellCheckModel().
      //setTensorflow(tf).
      readModel("../auxdata/spell_model", dataset.sparkSession, "").
      setInputCols(getOrDefault(inputCols))
  }

  def this() = this(Identifiable.randomUID("SPELL"))

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val requiredAnnotatorTypes: Array[String] = Array(AnnotatorType.TOKEN)
  override val annotatorType: AnnotatorType = AnnotatorType.TOKEN

  /* this is a list of functions that return the distance of a string to a particular regex */
  private var tokenClasses = List[(String) => Tuple2[String, Float]]()

}
