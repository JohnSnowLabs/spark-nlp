package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, TOKEN, WORD_EMBEDDINGS}
import com.johnsnowlabs.nlp.util.io.ReadAs
import com.johnsnowlabs.storage.Database.Name
import com.johnsnowlabs.storage.{Database, HasStorage, RocksDBConnection, StorageWriter}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.IntParam
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

/** Word Embeddings lookup annotator that maps tokens to vectors.
  * See [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/WordEmbeddingsTestSpec.scala]] for further reference on how to use this API.
  *
  * There are also two convenient functions to retrieve the embeddings coverage with respect to the transformed dataset:
  *
  * withCoverageColumn(dataset, embeddingsCol, outputCol): Adds a custom column with word coverage stats for the embedded field: (coveredWords, totalWords, coveragePercentage). This creates a new column with statistics for each row.
  *
  * overallCoverage(dataset, embeddingsCol): Calculates overall word coverage for the whole data in the embedded field. This returns a single coverage object considering all rows in the field.
  *
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
  **/
class WordEmbeddings(override val uid: String)
  extends AnnotatorApproach[WordEmbeddingsModel]
    with HasStorage
    with HasEmbeddingsProperties {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  def this() = this(Identifiable.randomUID("WORD_EMBEDDINGS"))

  /** Output annotation type : WORD_EMBEDDINGS
    *
    * @group anno
    **/
  override val outputAnnotatorType: AnnotatorType = WORD_EMBEDDINGS
  /** Output annotation type : DOCUMENT, TOKEN
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, TOKEN)

  /** Word Embeddings lookup annotator that maps tokens to vectors */
  override val description: String = "Word Embeddings lookup annotator that maps tokens to vectors"

  /** Error message */
  override protected val missingRefMsg: String = s"Please set storageRef param in $this. This ref is useful for other annotators" +
    " to require this particular set of embeddings. You can use any memorable name such as 'glove' or 'my_embeddings'."

  /** buffer size limit before dumping to disk storage while writing
    *
    * @group param
    **/
  val writeBufferSize = new IntParam(this, "writeBufferSize", "buffer size limit before dumping to disk storage while writing")
  setDefault(writeBufferSize, 10000)

  /** Buffer size limit before dumping to disk storage while writing.
    *
    * @group setParam
    **/
  def setWriteBufferSize(value: Int): this.type = set(writeBufferSize, value)

  /** cache size for items retrieved from storage. Increase for performance but higher memory consumption
    *
    * @group param
    **/
  val readCacheSize = new IntParam(this, "readCacheSize", "cache size for items retrieved from storage. Increase for performance but higher memory consumption")

  /** Cache size for items retrieved from storage. Increase for performance but higher memory consumption.
    *
    * @group setParam
    **/
  def setReadCacheSize(value: Int): this.type = set(readCacheSize, value)

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): WordEmbeddingsModel = {
    val model = new WordEmbeddingsModel()
      .setInputCols($(inputCols))
      .setStorageRef($(storageRef))
      .setDimension($(dimension))
      .setCaseSensitive($(caseSensitive))

    if (isSet(readCacheSize))
      model.setReadCacheSize($(readCacheSize))

    model
  }

  override protected def index(
                                fitDataset: Dataset[_],
                                storageSourcePath: Option[String],
                                readAs: Option[ReadAs.Value],
                                writers: Map[Database.Name, StorageWriter[_]],
                                readOptions: Option[Map[String, String]]
                              ): Unit = {
    val writer = writers.values.headOption
      .getOrElse(throw new IllegalArgumentException("Received empty WordEmbeddingsWriter from locators"))
      .asInstanceOf[WordEmbeddingsWriter]

    if (readAs.get == ReadAs.TEXT) {
      WordEmbeddingsTextIndexer.index(storageSourcePath.get, writer)
    }
    else if (readAs.get == ReadAs.BINARY) {
      WordEmbeddingsBinaryIndexer.index(storageSourcePath.get, writer)
    }
    else
      throw new IllegalArgumentException("Invalid WordEmbeddings read format. Must be either TEXT or BINARY")
  }

  override val databases: Array[Database.Name] = WordEmbeddingsModel.databases

  override protected def createWriter(database: Name, connection: RocksDBConnection): StorageWriter[_] = {
    new WordEmbeddingsWriter(connection, $(caseSensitive), $(dimension), get(readCacheSize).getOrElse(5000), $(writeBufferSize))
  }
}

object WordEmbeddings extends DefaultParamsReadable[WordEmbeddings]
