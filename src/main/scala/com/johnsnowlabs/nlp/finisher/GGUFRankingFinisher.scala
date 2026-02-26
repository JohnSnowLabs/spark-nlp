/*
 * Copyright 2017-2024 John Snow Labs
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
package com.johnsnowlabs.nlp.finisher

import com.johnsnowlabs.nlp.util.FinisherUtil
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.expressions.{UserDefinedFunction, Window}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}

/** Finisher for AutoGGUFReranker outputs that provides ranking capabilities including top-k
  * selection, sorting by relevance score, and score normalization.
  *
  * This finisher processes the output of AutoGGUFReranker, which contains documents with
  * relevance scores in their metadata. It provides several options for post-processing:
  *
  *   - Top-k selection: Select only the top k documents by relevance score
  *   - Score thresholding: Filter documents by minimum relevance score
  *   - Min-max scaling: Normalize relevance scores to 0-1 range
  *   - Sorting: Sort documents by relevance score in descending order
  *   - Ranking: Add rank information to document metadata
  *
  * The finisher preserves the document annotation structure while adding ranking information to
  * the metadata and optionally filtering/sorting the documents.
  *
  * ==Example==
  *
  * {{{
  * import com.johnsnowlabs.nlp.base._
  * import com.johnsnowlabs.nlp.annotators._
  * import com.johnsnowlabs.nlp.finisher._
  * import org.apache.spark.ml.Pipeline
  * import spark.implicits._
  *
  * val document = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val reranker = AutoGGUFReranker
  *   .pretrained("bge_reranker_v2_m3-Q4_K_M")
  *   .setInputCols("document")
  *   .setOutputCol("reranked_documents")
  *   .setQuery("A man is eating pasta.")
  *
  * val finisher = new GGUFRankingFinisher()
  *   .setInputCols("reranked_documents")
  *   .setOutputCol("ranked_documents")
  *   .setTopK(3)
  *   .setMinRelevanceScore(0.1)
  *   .setMinMaxScaling(true)
  *
  * val pipeline = new Pipeline().setStages(Array(document, reranker, finisher))
  *
  * val data = Seq(
  *   "A man is eating food.",
  *   "A man is eating a piece of bread.",
  *   "The girl is carrying a baby.",
  *   "A man is riding a horse."
  * ).toDF("text")
  *
  * val result = pipeline.fit(data).transform(data)
  * result.select("ranked_documents").show(truncate = false)
  * // Documents will be sorted by relevance with rank information in metadata
  * }}}
  *
  * @param uid
  *   required uid for storing finisher to disk
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param 1
  * @groupprio anno 2
  * @groupprio Ungrouped 3
  * @groupprio setParam 4
  * @groupprio getParam 5
  * @groupdesc param
  *   A list of (hyper-)parameter keys this finisher can take. Users can set and get the parameter
  *   values through setters and getters, respectively.
  */
case class GGUFRankingFinisher(override val uid: String)
    extends Transformer
    with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("GGUF_RANKING_FINISHER"))

  val RELEVANCE_SCORE_COL_NAME = "relevance_score"
  val QUERY_COL_NAME = "query"
  val RANK_COL_NAME = "rank"

  /** Name of input annotation cols containing reranked documents
    *
    * @group param
    */
  val inputCols: StringArrayParam =
    new StringArrayParam(
      this,
      "inputCols",
      "Name of input annotation cols containing reranked documents")

  /** Name of input annotation cols containing reranked documents
    *
    * @group setParam
    */
  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  /** Name of input annotation cols containing reranked documents
    *
    * @group setParam
    */
  def setInputCols(value: String*): this.type = setInputCols(value.toArray)

  /** Name of input annotation cols containing reranked documents
    *
    * @group getParam
    */
  def getInputCols: Array[String] = $(inputCols)

  /** Name of output annotation column containing ranked documents
    *
    * @group param
    */
  val outputCol: StringArrayParam =
    new StringArrayParam(
      this,
      "outputCol",
      "Name of output annotation column containing ranked documents")

  /** Name of output annotation column containing ranked documents
    *
    * @group setParam
    */
  def setOutputCol(value: String): this.type = set(outputCol, Array(value))

  /** Name of output annotation column containing ranked documents
    *
    * @group getParam
    */
  def getOutputCol: String = $(outputCol).headOption.getOrElse("ranked_documents")

  /** Maximum number of top documents to return based on relevance score
    *
    * @group param
    */
  val topK: IntParam =
    new IntParam(
      this,
      "topK",
      "Maximum number of top documents to return based on relevance score")

  /** Set maximum number of top documents to return
    *
    * @group setParam
    */
  def setTopK(value: Int): this.type = set(topK, value)

  /** Get maximum number of top documents to return
    *
    * @group getParam
    */
  def getTopK: Int = $(topK)

  /** Minimum relevance score threshold for filtering documents
    *
    * @group param
    */
  val minRelevanceScore: DoubleParam =
    new DoubleParam(
      this,
      "minRelevanceScore",
      "Minimum relevance score threshold for filtering documents")

  /** Set minimum relevance score threshold
    *
    * @group setParam
    */
  def setMinRelevanceScore(value: Double): this.type = set(minRelevanceScore, value)

  /** Get minimum relevance score threshold
    *
    * @group getParam
    */
  def getMinRelevanceScore: Double = $(minRelevanceScore)

  /** Whether to apply min-max scaling to normalize relevance scores to 0-1 range
    *
    * @group param
    */
  val minMaxScaling: BooleanParam =
    new BooleanParam(
      this,
      "minMaxScaling",
      "Whether to apply min-max scaling to normalize relevance scores to 0-1 range")

  /** Set whether to apply min-max scaling
    *
    * @group setParam
    */
  def setMinMaxScaling(value: Boolean): this.type = set(minMaxScaling, value)

  /** Get whether to apply min-max scaling
    *
    * @group getParam
    */
  def getMinMaxScaling: Boolean = $(minMaxScaling)

  setDefault(
    topK -> -1, // -1 means no limit
    minRelevanceScore -> Double.MinValue, // No threshold by default
    minMaxScaling -> false,
    outputCol -> Array("ranked_documents"))

  override def transform(dataset: Dataset[_]): DataFrame = {
    val inputCol = getInputCols.head
    val outputColumnName = getOutputCol

    import dataset.sparkSession.implicits._

    // First, flatten all annotations across all rows to get global statistics
    val allAnnotations = dataset
      .withColumn("row_id", monotonically_increasing_id())
      .select($"row_id", explode(col(inputCol)).as("annotation"))

    // Extract scores from all annotations
    val scoresDF = allAnnotations
      .withColumn(
        "score",
        when(
          col("annotation.metadata").getItem(RELEVANCE_SCORE_COL_NAME).isNotNull &&
            col("annotation.metadata").getItem(RELEVANCE_SCORE_COL_NAME) =!= "",
          col("annotation.metadata").getItem(RELEVANCE_SCORE_COL_NAME).cast("double"))
          .otherwise(0.0))

    // Get global min/max for scaling if enabled
    val (globalMin, globalMax) = if (getMinMaxScaling) {
      val stats = scoresDF.agg(min($"score"), max($"score")).collect().head
      (stats.getDouble(0), stats.getDouble(1))
    } else {
      (0.0, 0.0)
    }

    // Calculate scaled scores and assign global ranks
    val scaledDF = if (getMinMaxScaling && globalMax != globalMin) {
      scoresDF.withColumn("scaled_score", ($"score" - globalMin) / (globalMax - globalMin))
    } else if (getMinMaxScaling && globalMax == globalMin) {
      scoresDF.withColumn("scaled_score", lit(1.0))
    } else {
      scoresDF.withColumn("scaled_score", $"score")
    }

    // Filter by threshold and assign global ranks
    val filteredDF = scaledDF.filter($"scaled_score" >= getMinRelevanceScore)

    // Order by score and add row numbers using zipWithIndex to avoid Window partitioning
    val orderedDF = filteredDF.orderBy($"scaled_score".desc)
    val rankedRDD = orderedDF.rdd.zipWithIndex().map { case (row, index) =>
      Row.fromSeq(row.toSeq :+ (index + 1L))
    }

    // Create new schema with rank column
    val schemaWithRank =
      orderedDF.schema.add(StructField("global_rank", LongType, nullable = false))
    val rankedDF = orderedDF.sparkSession.createDataFrame(rankedRDD, schemaWithRank)

    // Apply top-k limit if specified
    val limitedDF = if (getTopK > 0) {
      rankedDF.filter($"global_rank" <= getTopK)
    } else {
      rankedDF
    }

    // Create UDF to update annotation metadata
    val updateAnnotationMetadataUDF: UserDefinedFunction = udf {
      (annotationRow: Row, score: Double, rank: Int) =>
        // Convert Row to Annotation
        val annotation = Annotation(
          annotatorType = annotationRow.getString(0),
          begin = annotationRow.getInt(1),
          end = annotationRow.getInt(2),
          result = annotationRow.getString(3),
          metadata = annotationRow.getAs[Map[String, String]](4))

        val updatedMetadata = annotation.metadata ++ Map(
          RELEVANCE_SCORE_COL_NAME -> score.toString,
          RANK_COL_NAME -> rank.toString)

        Annotation(
          annotatorType = annotation.annotatorType,
          begin = annotation.begin,
          end = annotation.end,
          result = annotation.result,
          metadata = updatedMetadata)
    }

    // Update annotations with new metadata
    val updatedAnnotationsDF = limitedDF
      .withColumn(
        "updated_annotation",
        updateAnnotationMetadataUDF($"annotation", $"scaled_score", $"global_rank"))
      .select($"row_id", $"updated_annotation")

    // Group annotations back by row_id
    val groupedDF = updatedAnnotationsDF
      .groupBy($"row_id")
      .agg(collect_list($"updated_annotation").as("processed_annotations"))

    // Join back with original dataset and filter out rows with no annotations
    val originalWithRowId = dataset.withColumn("row_id", monotonically_increasing_id())

    val result = originalWithRowId
      .join(
        groupedDF,
        Seq("row_id"),
        "inner"
      ) // Use inner join to exclude rows with no annotations
      .withColumn(outputColumnName, $"processed_annotations")
      .drop("row_id", "processed_annotations")

    result
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    val documentAnnotators = Seq(AnnotatorType.DOCUMENT)

    getInputCols.foreach { annotationColumn =>
      FinisherUtil.checkIfInputColsExist(getInputCols, schema)
      FinisherUtil.checkIfAnnotationColumnIsSparkNLPAnnotation(schema, annotationColumn)

      /** Check if the annotationColumn has Document type. It must be annotators that produce
        * Document annotations with relevance score metadata (like AutoGGUFReranker)
        */
      require(
        documentAnnotators.contains(schema(annotationColumn).metadata.getString("annotatorType")),
        s"column [$annotationColumn] must be of type Document with relevance score metadata")
    }

    // Add output column to schema
    val outputColumnName = getOutputCol
    schema.add(
      StructField(
        outputColumnName,
        ArrayType(Annotation.dataType),
        nullable = false,
        metadata = new MetadataBuilder()
          .putString("annotatorType", AnnotatorType.DOCUMENT)
          .build()))
  }
}

object GGUFRankingFinisher extends DefaultParamsReadable[GGUFRankingFinisher]
