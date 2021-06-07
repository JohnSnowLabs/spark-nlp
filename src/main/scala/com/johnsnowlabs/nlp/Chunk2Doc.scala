package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
  * Converts a `CHUNK` type column back into `DOCUMENT`. Useful when trying to re-tokenize or do further analysis on a
  * `CHUNK` result.
  *
  * For more extended examples on document pre-processing see the
  * [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/databricks_notebooks/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers_v3.0.ipynb Spark NLP Workshop]].
  *
  * ==Example==
  * Location entities are extracted and converted back into `DOCUMENT` type for further processing
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
  * import com.johnsnowlabs.nlp.Chunk2Doc
  *
  * val data = Seq((1, "New York and New Jersey aren't that far apart actually.")).toDF("id", "text")
  *
  * // Extracts Named Entities amongst other things
  * val pipeline = PretrainedPipeline("explain_document_dl")
  *
  * val chunkToDoc = new Chunk2Doc().setInputCols("entities").setOutputCol("chunkConverted")
  * val explainResult = pipeline.transform(data)
  *
  * val result = chunkToDoc.transform(explainResult)
  * result.selectExpr("explode(chunkConverted)").show(false)
  * +------------------------------------------------------------------------------+
  * |col                                                                           |
  * +------------------------------------------------------------------------------+
  * |[document, 0, 7, New York, [entity -> LOC, sentence -> 0, chunk -> 0], []]    |
  * |[document, 13, 22, New Jersey, [entity -> LOC, sentence -> 0, chunk -> 1], []]|
  * +------------------------------------------------------------------------------+
  * }}}
  *
  * @see [[com.johnsnowlabs.nlp.pretrained.PretrainedPipeline PretrainedPipeline]] on how to use the PretrainedPipeline
  * @see [[Doc2Chunk]] for converting `DOCUMENT` annotations to `CHUNK`
  * @param uid required uid for storing annotator to disk
  * @groupname anno Annotator types
  * @groupdesc anno Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc param A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  */
class Chunk2Doc(override val uid: String) extends AnnotatorModel[Chunk2Doc] with HasSimpleAnnotate[Chunk2Doc] {

  def this() = this(Identifiable.randomUID("CHUNK2DOC"))

  /**
    * Output annotator types: DOCUMENT
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  /**
    * Input annotator types: CHUNK
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(CHUNK)

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.map(annotation => {
      Annotation(
        outputAnnotatorType,
        annotation.begin,
        annotation.end,
        annotation.result,
        annotation.metadata
      )
    })
  }

}

object Chunk2Doc extends DefaultParamsReadable[Chunk2Doc]