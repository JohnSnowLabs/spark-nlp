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

package com.johnsnowlabs.ml.ai

import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, IMAGE, SENTENCE_EMBEDDINGS}
import com.johnsnowlabs.nlp.{Annotation, AnnotationImage, AnnotatorModel, HasBatchedAnnotate}
import com.johnsnowlabs.util.{ConfigHelper, ConfigLoader}
import org.apache.http.client.methods.{HttpGet, HttpPost}
import org.apache.http.entity.{ContentType, StringEntity}
import org.apache.http.impl.client.{CloseableHttpClient, HttpClients}
import org.apache.http.util.EntityUtils
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntParam, Param, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Dataset, Row, SparkSession}

import java.time.Instant
import java.util.UUID

/** Connector for storing and retrieving embeddings from vector databases.
  *
  * This annotator takes embeddings from previous annotators (like BertEmbeddings,
  * SentenceEmbeddings, E5VEmbeddings, etc.) and stores them in a vector database for similarity
  * search and retrieval. Currently supports Pinecone with more providers planned.
  *
  * Supports two modality modes:
  *   - 'text' (default): expects `DOCUMENT + SENTENCE_EMBEDDINGS` input columns. Metadata is
  *     augmented with `modality=text`.
  *   - 'image': expects `IMAGE + SENTENCE_EMBEDDINGS` input columns (e.g. from `ImageAssembler` +
  *     `E5VEmbeddings`). Metadata is augmented with `modality=image`, `image_origin`,
  *     `image_width`, `image_height`, and `image_nChannels`. Vector IDs are generated as
  *     deterministic UUID-v3 values derived from the image file path (origin), ensuring stable
  *     re-indexing.
  *
  * ==Supported Providers==
  *   - '''pinecone''': Pinecone vector database (default)
  *
  * ==Text Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.embeddings.BertSentenceEmbeddings
  * import com.johnsnowlabs.ml.ai.VectorDBConnector
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val embeddings = BertSentenceEmbeddings.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("sentence_embeddings")
  *
  * val vectorDB = new VectorDBConnector()
  *   .setInputCols("document", "sentence_embeddings")
  *   .setOutputCol("vectordb_result")
  *   .setProvider("pinecone")
  *   .setIndexName("my-index")
  *   .setNamespace("production")
  *   .setIdColumn("id")
  *   .setMetadataColumns(Array("text", "category"))
  *   .setBatchSize(100)
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   embeddings,
  *   vectorDB
  * ))
  *
  * val data = Seq(
  *   ("1", "Spark NLP is great", "tech"),
  *   ("2", "Vector databases enable semantic search", "tech")
  * ).toDF("id", "text", "category")
  *
  * val result = pipeline.fit(data).transform(data)
  * }}}
  *
  * ==Image Example==
  * {{{
  * import com.johnsnowlabs.nlp.base.ImageAssembler
  * import com.johnsnowlabs.nlp.embeddings.E5VEmbeddings
  * import com.johnsnowlabs.ml.ai.VectorDBConnector
  * import org.apache.spark.ml.Pipeline
  *
  * val imageAssembler = new ImageAssembler()
  *   .setInputCol("image")
  *   .setOutputCol("image_assembler")
  *
  * val e5vEmbeddings = E5VEmbeddings.pretrained()
  *   .setInputCols("image_assembler")
  *   .setOutputCol("image_embeddings")
  *
  * val vectorDB = new VectorDBConnector()
  *   .setInputCols("image_assembler", "image_embeddings")
  *   .setOutputCol("vectordb_result")
  *   .setProvider("pinecone")
  *   .setIndexName("my-multimodal-index")
  *   .setModalityMode("image")
  *   .setBatchSize(50)
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   imageAssembler,
  *   e5vEmbeddings,
  *   vectorDB
  * ))
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
  * @groupprio param 1
  * @groupprio anno 2
  * @groupprio Ungrouped 3
  * @groupprio setParam 4
  * @groupprio getParam 5
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class VectorDBConnector(override val uid: String)
    extends AnnotatorModel[VectorDBConnector]
    with HasBatchedAnnotate[VectorDBConnector] {

  def this() = this(Identifiable.randomUID("VECTOR_DB_CONNECTOR"))

  /** Input annotator types.
    *
    * In 'text' mode (default): `DOCUMENT, SENTENCE_EMBEDDINGS`
    *
    * In 'image' mode: `IMAGE, SENTENCE_EMBEDDINGS`
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, SENTENCE_EMBEDDINGS)
  override val outputAnnotatorType: String = DOCUMENT

  /** Modality mode for indexing: "text" (default) or "image".
    *
    * Set to "image" when the first input column contains IMAGE annotations (e.g. from
    * `ImageAssembler` + `E5VEmbeddings`). In image mode vector IDs are stable UUID-v3 values
    * derived from the image origin path and upserted records carry image-specific metadata
    * fields.
    *
    * @group param
    */
  val modalityMode = new Param[String](
    this,
    "modalityMode",
    "Modality mode for indexing. Supported values: 'text' (default), 'image'.")

  /** @group setParam */
  def setModalityMode(value: String): this.type = {
    require(
      Set("text", "image").contains(value),
      s"modalityMode must be 'text' or 'image', got: $value")
    set(modalityMode, value)
  }

  /** @group getParam */
  def getModalityMode: String = $(modalityMode)

  /** Override schema validation to accept IMAGE in the first input column when modalityMode is
    * set to "image", while still requiring SENTENCE_EMBEDDINGS in the second input column.
    */
  override protected def validate(schema: StructType): Boolean = {
    $(modalityMode) match {
      case "image" => checkSchema(schema, IMAGE) && checkSchema(schema, SENTENCE_EMBEDDINGS)
      case _ => checkSchema(schema, DOCUMENT) && checkSchema(schema, SENTENCE_EMBEDDINGS)
    }
  }

  /** Vector database provider (e.g., 'pinecone')
    * @group param
    */
  val provider = new Param[String](
    this,
    "provider",
    "Vector database provider. Currently supported: 'pinecone'")

  /** @group setParam */
  def setProvider(value: String): this.type = set(provider, value)

  /** @group getParam */
  def getProvider: String = $(provider)

  /** Index/collection name in the vector database
    * @group param
    */
  val indexName =
    new Param[String](this, "indexName", "Name of the index/collection in the vector database")

  /** @group setParam */
  def setIndexName(value: String): this.type = set(indexName, value)

  /** @group getParam */
  def getIndexName: String = $(indexName)

  /** Namespace/partition within the index (optional, provider-specific)
    * @group param
    */
  val namespace =
    new Param[String](this, "namespace", "Namespace/partition within the index (optional)")

  /** @group setParam */
  def setNamespace(value: String): this.type = set(namespace, value)

  /** @group getParam */
  def getNamespace: String = $(namespace)

  /** Column name to use as vector ID
    * @group param
    */
  val idColumn = new Param[String](
    this,
    "idColumn",
    "Column name to use as vector ID (if not set, generates UUID)")

  /** @group setParam */
  def setIdColumn(value: String): this.type = set(idColumn, value)

  /** @group getParam */
  def getIdColumn: String = $(idColumn)

  /** Metadata columns to include with vectors
    * @group param
    */
  val metadataColumns = new StringArrayParam(
    this,
    "metadataColumns",
    "Column names to include as metadata with vectors")

  /** @group setParam */
  def setMetadataColumns(value: Array[String]): this.type = set(metadataColumns, value)

  /** @group getParam */
  def getMetadataColumns: Array[String] = $(metadataColumns)

  /** Batch size for upsert operations
    * @group param
    */
  override val batchSize = new IntParam(
    this,
    "batchSize",
    "Number of vectors to upsert in a single batch",
    (x: Int) => x > 0 && x <= 1000)

  /** @group setParam */
  override def setBatchSize(value: Int): this.type = set(batchSize, value)

  /** @group getParam */
  override def getBatchSize: Int = $(batchSize)

  setDefault(
    provider -> "pinecone",
    batchSize -> 100,
    namespace -> "",
    metadataColumns -> Array.empty[String],
    modalityMode -> "text")

  // Provider-specific state
  private var apiKey: Option[Broadcast[String]] = None
  private var indexHost: Option[String] = None

  /** Set API key if not already set
    *
    * @param spark
    *   SparkSession
    * @param key
    *   API key for the vector database provider
    * @return
    *   this
    */
  def setApiKeyIfNotSet(spark: SparkSession, key: Option[String]): this.type = {
    if (apiKey.isEmpty && key.isDefined && key.get.nonEmpty) {
      apiKey = Some(spark.sparkContext.broadcast(key.get))
    }
    this
  }

  /** Get the broadcasted API key
    *
    * @return
    *   API key string
    */
  def getApiKey: String = {
    if (apiKey.isDefined) apiKey.get.value else ""
  }

  /** Initialize configuration before annotation
    *
    * Loads provider-specific configuration and initializes connection
    *
    * @param dataset
    *   Input dataset
    * @return
    *   Dataset unchanged
    */
  override def beforeAnnotate(dataset: Dataset[_]): Dataset[_] = {
    $(provider) match {
      case "pinecone" => initializePinecone(dataset)
      case other => throw new IllegalArgumentException(s"Unsupported provider: $other")
    }
    dataset
  }

  /** Initialize Pinecone-specific configuration
    *
    * @param dataset
    *   Input dataset for SparkSession access
    */
  private def initializePinecone(dataset: Dataset[_]): Unit = {
    // Load API key from config (universal vectordb key)
    this.setApiKeyIfNotSet(
      dataset.sparkSession,
      Some(ConfigLoader.getConfigStringValue(ConfigHelper.vectorDBAPIKey)))

    // Fetch index host from Pinecone API
    if (indexHost.isEmpty && isSet(indexName) && getApiKey.nonEmpty) {
      indexHost = fetchPineconeIndexHost($(indexName), getApiKey)
    }
  }

  /** Fetch the index host URL from Pinecone API
    *
    * @param indexName
    *   Name of the index
    * @param apiKey
    *   Pinecone API key
    * @return
    *   Optional host URL
    */
  private def fetchPineconeIndexHost(indexName: String, apiKey: String): Option[String] = {
    val url = s"https://api.pinecone.io/indexes/$indexName"
    val httpGet = new HttpGet(url)
    httpGet.setHeader("Api-Key", apiKey)
    httpGet.setHeader("X-Pinecone-API-Version", "2024-10")

    val httpclient: CloseableHttpClient = HttpClients.createDefault()
    try {
      val response = httpclient.execute(httpGet)
      val statusCode = response.getStatusLine.getStatusCode
      val responseBody = EntityUtils.toString(response.getEntity)

      if (statusCode != 200) {
        throw new Exception(
          s"Failed to fetch Pinecone index host. Status: $statusCode, Response: $responseBody")
      }

      val hostPattern = """"host"\s*:\s*"([^"]+)"""".r
      hostPattern.findFirstMatchIn(responseBody).map { m =>
        val host = m.group(1)
        if (host.startsWith("https://")) host else s"https://$host"
      }
    } catch {
      case ex: Exception if !ex.getMessage.startsWith("Failed to fetch") =>
        throw new Exception(s"Error fetching Pinecone index host: ${ex.getMessage}", ex)
    } finally {
      httpclient.close()
    }
  }

  /** Override batchProcess to get direct access to Row data.
    *
    * This is necessary because we need access to additional columns (idColumn, metadataColumns)
    * beyond just the annotations. The standard batchAnnotate only receives annotations.
    *
    * Branches on `modalityMode`: in "text" mode the first input column is expected to carry
    * DOCUMENT annotations; in "image" mode it carries IMAGE annotations (AnnotationImage).
    *
    * @param rows
    *   Iterator of rows to process
    * @return
    *   Iterator of processed rows with output annotation
    */
  override def batchProcess(rows: Iterator[_]): Iterator[Row] = {
    val allRows = rows.collect { case row: Row => row }.toSeq

    if (allRows.isEmpty) {
      return Iterator.empty
    }

    val processedBatches = allRows.grouped($(batchSize)).flatMap { batchRows =>
      val vectorsWithRows = batchRows.flatMap { row =>
        try {
          $(modalityMode) match {
            case "image" => processImageRow(row)
            case _ => processTextRow(row)
          }
        } catch {
          case e: Exception =>
            println(s"[VectorDBConnector] Error processing row: ${e.getMessage}")
            Some((row, null, Seq.empty[Annotation]))
        }
      }

      val vectors = vectorsWithRows.collect { case (_, v, _) if v != null => v }

      // Track whether the batch upsert actually succeeded on the remote provider.
      // A failed upsert must be reflected in the output annotation so callers are not
      // misled into thinking vectors were stored when they were not.
      var upsertError: Option[String] = None
      if (vectors.nonEmpty) {
        try {
          $(provider) match {
            case "pinecone" => upsertToPinecone(vectors)
            case other => throw new IllegalArgumentException(s"Unsupported provider: $other")
          }
        } catch {
          case e: Exception =>
            upsertError = Some(e.getMessage)
            println(s"[VectorDBConnector] Error upserting batch: ${e.getMessage}")
        }
      }

      vectorsWithRows.map { case (row, vector, docAnnotations) =>
        val outputAnnotations = if (vector != null) {
          val status = upsertError match {
            case None => "upserted"
            case Some(err) => s"failed: $err"
          }
          docAnnotations.map { ann =>
            Annotation(
              annotatorType = outputAnnotatorType,
              begin = ann.begin,
              end = ann.end,
              result = vector("id").asInstanceOf[String],
              metadata =
                ann.metadata + ("vectordb_status" -> status) + ("provider" -> $(provider)))
          }
        } else {
          Seq.empty[Annotation]
        }
        Row.fromSeq(row.toSeq :+ outputAnnotations.map(a => Row(a.productIterator.toSeq: _*)))
      }
    }

    processedBatches
  }

  /** Required by HasBatchedAnnotate trait - but we override batchProcess instead since we need
    * Row-level access for idColumn and metadataColumns.
    */
  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {
    batchedAnnotations.map(_ => Seq.empty[Annotation])
  }

  /** Build and return a vector payload for a text-mode row (DOCUMENT + SENTENCE_EMBEDDINGS).
    *
    * Metadata always contains `modality=text` in addition to any user-specified columns.
    */
  private def processTextRow(row: Row): Option[(Row, Map[String, Any], Seq[Annotation])] = {
    val embeddingAnnotations = getAnnotationsFromRow(row, getInputCols.last)
    val documentAnnotations = getAnnotationsFromRow(row, getInputCols.head)
    if (embeddingAnnotations.nonEmpty) {
      val embeddings = embeddingAnnotations.head.embeddings
      val vectorId = resolveVectorId(row, None)
      val metadata: Map[String, String] =
        resolveUserMetadata(row) + ("modality" -> "text") + ("indexed_at" -> Instant.now().toString)
      val vector: Map[String, Any] =
        Map("id" -> vectorId, "values" -> embeddings.toList, "metadata" -> metadata)
      Some((row, vector, documentAnnotations))
    } else {
      Some((row, null, documentAnnotations))
    }
  }

  /** Build and return a vector payload for an image-mode row (IMAGE + SENTENCE_EMBEDDINGS).
    *
    * Metadata always contains `modality=image`, `image_origin`, `image_width`, `image_height`,
    * and `image_nChannels` in addition to any user-specified columns. When no explicit `idColumn`
    * is configured the vector ID is a deterministic UUID-v3 derived from the image origin path,
    * ensuring stable re-indexing of the same asset.
    *
    * Synthetic DOCUMENT output annotations are produced from the IMAGE annotations so that
    * downstream pipeline stages receive valid annotation rows. Each synthesised annotation
    * carries the image metadata so callers can inspect it after the transform.
    */
  private def processImageRow(row: Row): Option[(Row, Map[String, Any], Seq[Annotation])] = {
    val embeddingAnnotations = getAnnotationsFromRow(row, getInputCols.last)
    val imageAnnotations = getImageAnnotationsFromRow(row, getInputCols.head)

    // Synthesise output annotations regardless of whether we have embeddings so that empty
    // rows still produce a well-formed output column.
    val syntheticAnns: Seq[Annotation] = imageAnnotations.map { imgAnn =>
      Annotation(
        annotatorType = outputAnnotatorType,
        begin = 0,
        end = 0,
        result = imgAnn.origin,
        metadata = Map(
          "modality" -> "image",
          "image_origin" -> imgAnn.origin,
          "image_filename" -> extractFilename(imgAnn.origin),
          "image_width" -> imgAnn.width.toString,
          "image_height" -> imgAnn.height.toString,
          "image_nChannels" -> imgAnn.nChannels.toString))
    }

    if (embeddingAnnotations.nonEmpty && imageAnnotations.nonEmpty) {
      val embeddings = embeddingAnnotations.head.embeddings
      val firstImage = imageAnnotations.head
      val vectorId = resolveVectorId(row, Some(firstImage.origin))
      val imageMeta: Map[String, String] = Map(
        "modality" -> "image",
        "image_origin" -> firstImage.origin,
        "image_filename" -> extractFilename(firstImage.origin),
        "image_width" -> firstImage.width.toString,
        "image_height" -> firstImage.height.toString,
        "image_nChannels" -> firstImage.nChannels.toString,
        "indexed_at" -> Instant.now().toString)
      val metadata: Map[String, String] = resolveUserMetadata(row) ++ imageMeta
      val vector: Map[String, Any] =
        Map("id" -> vectorId, "values" -> embeddings.toList, "metadata" -> metadata)
      Some((row, vector, syntheticAnns))
    } else {
      Some((row, null, syntheticAnns))
    }
  }

  /** Resolve the vector ID for a row.
    *
    * Priority:
    *   1. Value from `idColumn` DataFrame column (if set and present). 2. Stable UUID-v3 derived
    *      from `fallbackOrigin` (used for image paths). 3. Random UUID.
    */
  private def resolveVectorId(row: Row, fallbackOrigin: Option[String]): String = {
    if (isSet(idColumn)) {
      try {
        val idIdx = row.fieldIndex($(idColumn))
        Option(row.get(idIdx))
          .map(_.toString)
          .getOrElse(fallbackOrigin.map(stableImageId).getOrElse(UUID.randomUUID().toString))
      } catch {
        case _: Exception =>
          fallbackOrigin.map(stableImageId).getOrElse(UUID.randomUUID().toString)
      }
    } else {
      fallbackOrigin.map(stableImageId).getOrElse(UUID.randomUUID().toString)
    }
  }

  /** Collect user-specified metadata columns from the row. Returns an empty map when
    * `metadataColumns` is not set.
    */
  private def resolveUserMetadata(row: Row): Map[String, String] = {
    if (isSet(metadataColumns) && $(metadataColumns).nonEmpty) {
      $(metadataColumns).flatMap { col =>
        try {
          val colIdx = row.fieldIndex(col)
          Option(row.get(colIdx)).map(v => col -> v.toString)
        } catch {
          case _: Exception => None
        }
      }.toMap
    } else {
      Map.empty[String, String]
    }
  }

  /** Generate a deterministic, stable UUID-v3 from an image origin path.
    *
    * Repeated calls with the same `origin` always produce the same ID, which makes re-indexing
    * idempotent without an explicit `idColumn`.
    */
  private def stableImageId(origin: String): String =
    UUID.nameUUIDFromBytes(origin.getBytes("UTF-8")).toString

  /** Extract the filename (last path segment) from an image origin URI or file-system path.
    *
    * Examples:
    *   - `"file:///content/test_images/cat.jpg"` → `"cat.jpg"`
    *   - `"s3://my-bucket/images/dog.png"` → `"dog.png"`
    *   - `"/local/path/photo.jpeg"` → `"photo.jpeg"`
    */
  private def extractFilename(origin: String): String = {
    val stripped = origin.replaceAll("^file://", "")
    val lastSlash = stripped.lastIndexOf('/')
    if (lastSlash >= 0 && lastSlash < stripped.length - 1) stripped.substring(lastSlash + 1)
    else stripped
  }

  /** Helper method to get Annotation items from a row (text / embeddings columns).
    *
    * @param row
    *   Input row
    * @param column
    *   Column name
    * @return
    *   Sequence of annotations
    */
  private def getAnnotationsFromRow(row: Row, column: String): Seq[Annotation] = {
    try {
      val colIdx = row.fieldIndex(column)
      val annotations = row.getAs[Seq[Row]](colIdx)
      if (annotations != null) annotations.map(Annotation(_)) else Seq.empty
    } catch {
      case _: Exception => Seq.empty[Annotation]
    }
  }

  /** Helper method to get AnnotationImage items from a row (IMAGE columns).
    *
    * @param row
    *   Input row
    * @param column
    *   Column name – must reference an IMAGE-typed annotation column
    * @return
    *   Sequence of AnnotationImage
    */
  private def getImageAnnotationsFromRow(row: Row, column: String): Seq[AnnotationImage] = {
    try {
      val colIdx = row.fieldIndex(column)
      val annotations = row.getAs[Seq[Row]](colIdx)
      if (annotations != null) annotations.map(AnnotationImage(_)) else Seq.empty
    } catch {
      case _: Exception => Seq.empty[AnnotationImage]
    }
  }

  /** Upsert vectors to Pinecone
    *
    * @param vectors
    *   Vectors to upsert
    */
  private def upsertToPinecone(vectors: Seq[Map[String, Any]]): Unit = {
    require(
      indexHost.isDefined,
      "Pinecone index host not initialized. Check API key and index name.")

    val url = s"${indexHost.get}/vectors/upsert"
    val httpPost = new HttpPost(url)

    val namespaceJson =
      if (isSet(namespace) && $(namespace).nonEmpty) s""""namespace": "${$(namespace)}","""
      else ""

    val jsonBody = s"""{
                      |  $namespaceJson
                      |  "vectors": ${vectorsToJson(vectors)}
                      |}""".stripMargin

    httpPost.setEntity(new StringEntity(jsonBody, ContentType.APPLICATION_JSON))

    val key = getApiKey
    require(key.nonEmpty, "Pinecone API Key required")
    httpPost.setHeader("Api-Key", key)
    httpPost.setHeader("Content-Type", "application/json")

    val httpclient: CloseableHttpClient = HttpClients.createDefault()
    try {
      val response = httpclient.execute(httpPost)
      val statusCode = response.getStatusLine.getStatusCode
      val responseBody = EntityUtils.toString(response.getEntity)

      if (statusCode != 200) {
        throw new Exception(s"Pinecone upsert failed with status $statusCode: $responseBody")
      }
    } catch {
      case ex: Exception if !ex.getMessage.startsWith("Pinecone upsert failed") =>
        throw new Exception(s"Error upserting to Pinecone: ${ex.getMessage}", ex)
    } finally {
      httpclient.close()
    }
  }

  /** Convert vectors to JSON string
    *
    * @param vectors
    *   Vectors to convert
    * @return
    *   JSON string
    */
  private def vectorsToJson(vectors: Seq[Map[String, Any]]): String = {
    val vectorsJson = vectors.map { vector =>
      val id = vector("id").asInstanceOf[String]
      val values = vector("values").asInstanceOf[List[Float]]
      val metadata = vector("metadata").asInstanceOf[Map[String, String]]

      val metadataJson = if (metadata.nonEmpty) {
        val metadataFields = metadata
          .map { case (k, v) =>
            s""""$k": "${escapeJsonString(v)}""""
          }
          .mkString(", ")
        s""", "metadata": { $metadataFields }"""
      } else {
        ""
      }

      val valuesJson = values.mkString(", ")
      s"""{"id": "${escapeJsonString(id)}", "values": [$valuesJson]$metadataJson}"""
    }

    s"[${vectorsJson.mkString(", ")}]"
  }

  /** Escape JSON special characters
    *
    * @param input
    *   Input string
    * @return
    *   Escaped string
    */
  private def escapeJsonString(input: String): String = {
    input.map {
      case '"' => "\\\""
      case '\\' => "\\\\"
      case '/' => "\\/"
      case '\b' => "\\b"
      case '\f' => "\\f"
      case '\n' => "\\n"
      case '\r' => "\\r"
      case '\t' => "\\t"
      case c => c
    }.mkString
  }

}

/** This is the companion object of [[VectorDBConnector]]. Please refer to that class for the
  * documentation.
  */
object VectorDBConnector extends DefaultParamsReadable[VectorDBConnector]
