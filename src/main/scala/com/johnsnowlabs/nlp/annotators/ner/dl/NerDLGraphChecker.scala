/*
 * Copyright 2017-2025 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators.ner.dl

import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.embeddings.HasEmbeddingsProperties
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.param.{IntParam, Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.StructType

import scala.util.{Failure, Success, Try}

/** Checks whether a suitable NerDL graph is available for the given training dataset, before any
  * computations/training is done. This annotator is useful for custom training cases, where
  * specialized graphs might not be available and we want to check before embeddings are
  * evaluated.
  *
  * Important: This annotator should be used or positioned before any embedding or NerDLApproach
  * annotators in the pipeline and will process the whole dataset to extract the required graph
  * parameters.
  *
  * This annotator requires a dataset with at least two columns: one with tokens and one with the
  * labels. In addition, it requires the used embedding annotator in the pipeline to extract the
  * suitable embedding dimension.
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master//home/ducha/Workspace/scala/spark-nlp-feature/examples/python/training/english/dl-ner/ner_dl_graph_checker.ipynb Examples]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/ner/dl/NerDLGraphCheckerTestSpec.scala NerDLGraphCheckerTestSpec]].
  *
  * ==Example==
  * {{{
  * import com.johnsnowlabs.nlp.annotator._
  * import com.johnsnowlabs.nlp.training.CoNLL
  * import org.apache.spark.ml.Pipeline
  *
  * // This CoNLL dataset already includes a sentence, token and label
  * // column with their respective annotator types. If a custom dataset is used,
  * // these need to be defined with for example:
  * val conll = CoNLL()
  * val trainingData = conll.readDataset(spark, "src/test/resources/conll2003/eng.train")
  *
  * val embeddings = BertEmbeddings
  *   .pretrained()
  *   .setInputCols("sentence", "token")
  *   .setOutputCol("embeddings")
  *
  * // Requires the data for NerDLApproach graphs: text, tokens, labels and the embedding model
  * val nerDLGraphChecker = new NerDLGraphChecker()
  *   .setInputCols("sentence", "token")
  *   .setLabelColumn("label")
  *   .setEmbeddingsModel(embeddings)
  *
  * val nerTagger = new NerDLApproach()
  *   .setInputCols("sentence", "token", "embeddings")
  *   .setLabelColumn("label")
  *   .setOutputCol("ner")
  *   .setMaxEpochs(1)
  *   .setRandomSeed(0)
  *   .setVerbose(0)
  *
  * val pipeline = new Pipeline().setStages(
  *   Array(nerDLGraphChecker, embeddings, nerTagger))
  *
  * // Will throw an exception if no suitable graph is found
  * val pipelineModel = pipeline.fit(trainingData)
  * }}}
  *
  * @param uid
  *   required uid for storing annotator to disk
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
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
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class NerDLGraphChecker(override val uid: String)
    extends Estimator[NerDLGraphCheckerModel]
    with HasInputAnnotationCols
    with ParamsAndFeaturesWritable {
  import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.implicits._

  def this() = this(Identifiable.randomUID("NerDLGraphChecker"))

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  override val inputAnnotatorTypes: Array[String] =
    Array(AnnotatorType.DOCUMENT, AnnotatorType.TOKEN)

  /** Column with label per each token
    *
    * @group param
    */
  val labelColumn = new Param[String](this, "labelColumn", "Column with label per each token")

  /** @group setParam */
  def setLabelColumn(value: String): this.type = set(labelColumn, value)

  /** @group getParam */
  def getLabelColumn: String = $(labelColumn)

  /** Dimensionality of embeddings
    *
    * @group param
    */
  val embeddingsDim = new IntParam(this, "embeddingsDim", "Dimensionality of embeddings")

  /** @group setParam */
  def setEmbeddingsModel(model: AnnotatorModel[_] with HasEmbeddingsProperties): this.type = {
    val dim = Try {
      model.getDimension
    } match {
      case Failure(exception) =>
        throw new IllegalArgumentException(
          s"Embeddings dimension could not be inferred from the provided embeddings model. " +
            s"Please set it manually with .setEmbeddingsDim(). Error: ${exception.getMessage}")
      case Success(value) => value
    }
    setEmbeddingsDim(dim)
  }

  /** @group getParam */
  def getEmbeddingsDim: Int = $(embeddingsDim)

  /** @group setParam */
  def setEmbeddingsDim(d: Int): this.type = set(embeddingsDim, d)

  /** Folder path that contain external graph files
    *
    * @group param
    */
  val graphFolder =
    new Param[String](this, "graphFolder", "Folder path that contain external graph files")

  /** @group getParam */
  protected def getGraphFolder: Option[String] = get(graphFolder)

  /** Extracts the graph hyperparameters from the training data (dataset).
    *
    * * @param dataset the training dataset
    * @param inputCols
    *   the input columns that contain the tokens and embeddings
    * @param labelsCol
    *   the column that contains the labels
    * @throws IllegalArgumentException
    *   if the token input column is not found in the dataset schema*
    * @return
    *   a tuple containing the number of labels, number of unique characters, and the embedding
    *   dim
    */
  protected def getGraphParamsDs(
      dataset: Dataset[_],
      inputCols: Array[String],
      labelsCol: String): (Int, Int, Int) = {
    def getCol(annoType: String) = {
      dataset.schema.fields.find { field =>
        inputCols.contains(field.name) && field.metadata.getString("annotatorType") == annoType
      } match {
        case Some(value) => col(value.name)
        case None =>
          new IllegalArgumentException(s"Token input column not found in the dataset schema.")
          col("")
      }
    }

    val tokenCol: String = getCol(AnnotatorType.TOKEN).toString

    val nLabels = dataset
      .select(labelsCol)
      .map(r => Annotation.getAnnotations(r, labelsCol))
      .flatMap { annotations: Seq[Annotation] =>
        annotations.map(_.result)
      }
      .distinct()
      .count()
      .toInt

    val nChars: Int = dataset
      .select(tokenCol)
      .map(r => Annotation.getAnnotations(r, tokenCol))
      .flatMap { annotations: Seq[Annotation] =>
        annotations.flatMap(_.result.toArray.map(_.toString))
      }
      .distinct()
      .count()
      .toInt

    val embeddingsDim = getEmbeddingsDim

    (nLabels, nChars, embeddingsDim)
  }

  protected def searchForSuitableGraph(nLabels: Int, nChars: Int, embeddingsDim: Int): String =
    NerDLApproach.searchForSuitableGraph(nLabels, embeddingsDim, nChars + 1, getGraphFolder)

  override def fit(dataset: Dataset[_]): NerDLGraphCheckerModel = {
    val (nLabels, nChars, embeddingsDim) =
      getGraphParamsDs(dataset, $(inputCols), $(labelColumn))

    // Throws exception if no suitable graph found
    Try {
      searchForSuitableGraph(nLabels, nChars, embeddingsDim)
    } match {
      case Failure(exception: IllegalArgumentException) =>
        throw new IllegalArgumentException("NerDLGraphChecker: " + exception.getMessage)
      case Failure(exception) => throw exception
      case Success(name) =>
        val graphPrefixRegex = "^.+blstm_"
        val clean = name.replaceFirst(graphPrefixRegex, "").replace(".pb", "")
        val graphParams = clean.split("_").take(4).map(s => s.toInt)
        val Array(fileTags, fileEmbeddingsNDims, _, fileNChars) = graphParams
        println(
          s"NerDLGraphChecker: For dataset embeddings dim: $embeddingsDim tags: $nLabels nChars: ${nChars + 1}" +
            s" found suitable graph with embeddings dim: $fileEmbeddingsNDims tags: $fileTags nChars: $fileNChars.")
    }

    new NerDLGraphCheckerModel()
      .setInputCols(getInputCols)
      .setLabelColumn(getLabelColumn)
      .setEmbeddingsDim(embeddingsDim)
  }

  override def transformSchema(schema: StructType): StructType = schema

  override def copy(extra: ParamMap): Estimator[NerDLGraphCheckerModel] = defaultCopy(extra)
}

object NerDLGraphChecker extends ParamsAndFeaturesReadable[NerDLGraphChecker]
