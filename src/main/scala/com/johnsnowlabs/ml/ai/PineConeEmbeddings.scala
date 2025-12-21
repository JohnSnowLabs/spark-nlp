package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, SENTENCE_EMBEDDINGS}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasBatchedAnnotate}
import com.johnsnowlabs.util.{ConfigHelper, ConfigLoader}
import org.apache.http.client.methods.HttpPost
import org.apache.http.entity.{ContentType, StringEntity}
import org.apache.http.impl.client.{CloseableHttpClient, HttpClients}
import org.apache.http.util.EntityUtils
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntParam, Param, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.{Dataset, Row, SparkSession}

import java.util.UUID

/** Stores embeddings into Pinecone vector database.
 *
 * This annotator takes embeddings from previous annotators (like BertEmbeddings,
 * SentenceEmbeddings, OpenAIEmbeddings, etc.) and stores them in Pinecone for similarity search
 * and retrieval.
 *
 * Pinecone is a managed vector database that enables fast similarity search at scale.
 *
 * ==Example==
 * {{{
 * import spark.implicits._
 * import com.johnsnowlabs.nlp.base.DocumentAssembler
 * import com.johnsnowlabs.nlp.embeddings.BertEmbeddings
 * import com.johnsnowlabs.nlp.annotators.PineconeEmbeddings
 * import org.apache.spark.ml.Pipeline
 *
 * val documentAssembler = new DocumentAssembler()
 *   .setInputCol("text")
 *   .setOutputCol("document")
 *
 * val embeddings = BertEmbeddings.pretrained()
 *   .setInputCols("document")
 *   .setOutputCol("embeddings")
 *
 * val pinecone = new PineconeEmbeddings()
 *   .setInputCols("document", "embeddings")
 *   .setOutputCol("pinecone_result")
 *   .setIndexName("my-index")
 *   .setNamespace("production")
 *   .setIdColumn("id")
 *   .setMetadataColumns(Array("text", "category"))
 *   .setBatchSize(100)
 *
 * val pipeline = new Pipeline().setStages(Array(
 *   documentAssembler,
 *   embeddings,
 *   pinecone
 * ))
 *
 * val data = Seq(
 *   ("1", "Spark NLP is great", "tech"),
 *   ("2", "Pinecone is a vector database", "tech")
 * ).toDF("id", "text", "category")
 *
 * val result = pipeline.fit(data).transform(data)
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
class PineconeEmbeddings(override val uid: String)
  extends AnnotatorModel[PineconeEmbeddings]
    with HasBatchedAnnotate[PineconeEmbeddings] {

  def this() = this(Identifiable.randomUID("PINECONE_EMBEDDINGS"))

  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, SENTENCE_EMBEDDINGS)
  override val outputAnnotatorType: String = DOCUMENT

  /** Pinecone index name
   * @group param
   */
  val indexName = new Param[String](
    this,
    "indexName",
    "Name of the Pinecone index to store vectors")

  /** @group setParam */
  def setIndexName(value: String): this.type = set(indexName, value)

  /** @group getParam */
  def getIndexName: String = $(indexName)

  /** Pinecone namespace (optional)
   * @group param
   */
  val namespace = new Param[String](
    this,
    "namespace",
    "Namespace within the Pinecone index (optional)")

  /** @group setParam */
  def setNamespace(value: String): this.type = set(namespace, value)

  /** @group getParam */
  def getNamespace: String = $(namespace)

  /** Pinecone environment
   * @group param
   */
  val environment = new Param[String](
    this,
    "environment",
    "Pinecone environment (e.g., 'us-east-1-aws')")

  /** @group setParam */
  def setEnvironment(value: String): this.type = set(environment, value)

  /** @group getParam */
  def getEnvironment: String = $(environment)

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

  /** Metadata columns to include
   * @group param
   */
  val metadataColumns = new StringArrayParam(
    this,
    "metadataColumns",
    "Column names to include as metadata in Pinecone")

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
    batchSize -> 100,
    namespace -> "",
    metadataColumns -> Array.empty[String])

  private var apiKey: Option[Broadcast[String]] = None
  private var indexHost: Option[String] = None

  /** Set API key if not already set
   *
   * @param spark
   *   SparkSession
   * @param pineconeKey
   *   Pinecone API key
   * @return
   *   this
   */
  def setApiKeyIfNotSet(spark: SparkSession, pineconeKey: Option[String]): this.type = {
    if (apiKey.isEmpty && pineconeKey.isDefined) {
      apiKey = Some(spark.sparkContext.broadcast(pineconeKey.get))
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
   * Loads API key from configuration and initializes index host
   *
   * @param dataset
   *   Input dataset
   * @return
   *   Dataset unchanged
   */
  override def beforeAnnotate(dataset: Dataset[_]): Dataset[_] = {
    // Load API key from config (checks environment variables, system properties, and config file)
    this.setApiKeyIfNotSet(
      dataset.sparkSession,
      Some(ConfigLoader.getConfigStringValue(ConfigHelper.pineconeAPIKey)))

    // Load environment from config if not explicitly set
    if (!isSet(environment)) {
      val configEnv = ConfigLoader.getConfigStringValue(ConfigHelper.pineconeEnvironment)
      if (configEnv.nonEmpty) {
        set(environment, configEnv)
      }
    }

    if (!isSet(indexName)) {
      val configIndex = ConfigLoader.getConfigStringValue(ConfigHelper.pineconeIndexName)
      if (configIndex.nonEmpty) {
        set(indexName, configIndex)
      }
    }

    // Initialize index host by fetching from Pinecone API
    if (indexHost.isEmpty && isSet(indexName) && getApiKey.nonEmpty) {
      indexHost = fetchIndexHost($(indexName), getApiKey)
    }

    dataset
  }

  /** Override batchProcess to get direct access to Row data.
   *
   * This is necessary because we need access to additional columns (idColumn, metadataColumns)
   * beyond just the annotations. The standard batchAnnotate only receives annotations.
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

    // Process in batches for Pinecone API efficiency
    val processedBatches = allRows.grouped($(batchSize)).flatMap { batchRows =>
      val vectorsWithRows = batchRows.flatMap { row =>
        try {
          // Get annotations from input columns
          val embeddingAnnotations = getAnnotationsFromRow(row, getInputCols.last)
          val documentAnnotations = getAnnotationsFromRow(row, getInputCols.head)

          if (embeddingAnnotations.nonEmpty) {
            val embeddings = embeddingAnnotations.head.embeddings

            // Extract vector ID from row column or generate UUID
            val vectorId = if (isSet(idColumn)) {
              try {
                val idIdx = row.fieldIndex($(idColumn))
                row.get(idIdx).toString
              } catch {
                case _: Exception => UUID.randomUUID().toString
              }
            } else {
              UUID.randomUUID().toString
            }

            // Extract metadata from specified columns
            val vectorMetadata = if (isSet(metadataColumns) && $(metadataColumns).nonEmpty) {
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

            val vector = Map(
              "id" -> vectorId,
              "values" -> embeddings.toList,
              "metadata" -> vectorMetadata)

            Some((row, vector, documentAnnotations))
          } else {
            Some((row, null, documentAnnotations))
          }
        } catch {
          case e: Exception =>
            println(s"[PineconeEmbeddings] Error processing row: ${e.getMessage}")
            Some((row, null, Seq.empty[Annotation]))
        }
      }

      // Collect non-null vectors for batch upsert
      val vectors = vectorsWithRows.collect { case (_, v, _) if v != null => v }

      // Upsert batch to Pinecone
      if (vectors.nonEmpty) {
        try {
          upsertVectors(vectors)
        } catch {
          case e: Exception =>
            println(s"[PineconeEmbeddings] Error upserting batch: ${e.getMessage}")
        }
      }

      // Return rows with output annotations
      vectorsWithRows.map { case (row, vector, docAnnotations) =>
        val outputAnnotations = if (vector != null) {
          // Create result annotation indicating success
          docAnnotations.map { ann =>
            Annotation(
              annotatorType = outputAnnotatorType,
              begin = ann.begin,
              end = ann.end,
              result = vector("id").asInstanceOf[String],
              metadata = ann.metadata + ("pinecone_status" -> "upserted")
            )
          }
        } else {
          Seq.empty[Annotation]
        }
        Row.fromSeq(row.toSeq :+ outputAnnotations.map(a => Row(a.productIterator.toSeq: _*)))
      }
    }

    processedBatches
  }

  /** Required by HasBatchedAnnotate trait - but we override batchProcess instead
   * since we need Row-level access for idColumn and metadataColumns.
   */
  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {
    // This method is not used since we override batchProcess directly
    // Return empty annotations - the actual work happens in batchProcess
    batchedAnnotations.map(_ => Seq.empty[Annotation])
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
  private def fetchIndexHost(indexName: String, apiKey: String): Option[String] = {
    import org.apache.http.client.methods.HttpGet

    val url = s"https://api.pinecone.io/indexes/$indexName"
    val httpGet = new HttpGet(url)
    httpGet.setHeader("Api-Key", apiKey)
    httpGet.setHeader("X-Pinecone-API-Version", "2024-10")

    val httpclient: CloseableHttpClient = HttpClients.createDefault()
    try {
      val response = httpclient.execute(httpGet)
      val statusCode = response.getStatusLine.getStatusCode
      val responseBody = EntityUtils.toString(response.getEntity)

      if (statusCode == 200) {
        // Parse host from JSON response
        val hostPattern = """"host"\s*:\s*"([^"]+)"""".r
        hostPattern.findFirstMatchIn(responseBody).map { m =>
          val host = m.group(1)
          if (host.startsWith("https://")) host else s"https://$host"
        }
      } else {
        println(s"[PineconeEmbeddings] Failed to fetch index host: $statusCode - $responseBody")
        None
      }
    } catch {
      case ex: Exception =>
        println(s"[PineconeEmbeddings] Error fetching index host: ${ex.getMessage}")
        None
    } finally {
      httpclient.close()
    }
  }

  /** Helper method to get annotations from a row
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

  /** Upsert vectors to Pinecone
   *
   * @param vectors
   *   Vectors to upsert
   */
  private def upsertVectors(vectors: Seq[Map[String, Any]]): Unit = {
    require(indexHost.isDefined, "Index host not initialized. Ensure environment is set.")

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
      case ex: Exception =>
        println(s"[PineconeEmbeddings] Exception during upsert: ${ex.getMessage}")
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
        val metadataFields = metadata.map { case (k, v) =>
          s""""$k": "${escapeJsonString(v)}""""
        }.mkString(", ")
        s""", "metadata": { $metadataFields }"""
      } else {
        ""
      }

      val valuesJson = values.mkString(", ")

      s"""{"id": "$id", "values": [$valuesJson]$metadataJson}"""
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
    input
      .map {
        case '"' => "\\\""
        case '\\' => "\\\\"
        case '/' => "\\/"
        case '\b' => "\\b"
        case '\f' => "\\f"
        case '\n' => "\\n"
        case '\r' => "\\r"
        case '\t' => "\\t"
        case c => c
      }
      .mkString
  }

  /** Query Pinecone for similar vectors
   *
   * @param queryVector
   *   Query vector
   * @param topK
   *   Number of results to return
   * @param includeMetadata
   *   Whether to include metadata
   * @return
   *   Query results
   */
  def query(
             queryVector: Array[Float],
             topK: Int = 10,
             includeMetadata: Boolean = true): String = {
    require(indexHost.isDefined, "Index host not initialized. Ensure environment is set.")

    val url = s"${indexHost.get}/query"
    val httpPost = new HttpPost(url)

    val namespaceJson =
      if (isSet(namespace) && $(namespace).nonEmpty) s""""namespace": "${$(namespace)}","""
      else ""

    val valuesJson = queryVector.mkString(", ")

    val jsonBody = s"""{
                      |  $namespaceJson
                      |  "topK": $topK,
                      |  "includeMetadata": $includeMetadata,
                      |  "vector": [$valuesJson]
                      |}""".stripMargin

    httpPost.setEntity(new StringEntity(jsonBody, ContentType.APPLICATION_JSON))

    val key = getApiKey
    require(key.nonEmpty, "Pinecone API Key required")
    httpPost.setHeader("Api-Key", key)
    httpPost.setHeader("Content-Type", "application/json")

    var responseBody: String = ""
    val httpclient: CloseableHttpClient = HttpClients.createDefault()
    try {
      val response = httpclient.execute(httpPost)
      val statusCode = response.getStatusLine.getStatusCode
      responseBody = EntityUtils.toString(response.getEntity)

      if (statusCode != 200) {
        throw new Exception(s"Pinecone query failed with status $statusCode: $responseBody")
      }
    } catch {
      case ex: Exception =>
        throw new Exception(s"Error querying Pinecone: ${ex.getMessage}", ex)
    } finally {
      httpclient.close()
    }

    responseBody
  }
}

/** This is the companion object of [[PineconeEmbeddings]]. Please refer to that class for the
 * documentation.
 */
object PineconeEmbeddings extends DefaultParamsReadable[PineconeEmbeddings]