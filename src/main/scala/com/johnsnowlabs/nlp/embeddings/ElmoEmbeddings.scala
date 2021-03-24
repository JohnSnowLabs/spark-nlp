package com.johnsnowlabs.nlp.embeddings

import java.io.File

import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, IntParam, Param}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}

/** Embeddings from a language model trained on the 1 Billion Word Benchmark.
  *
  * Note that this is a very computationally expensive module compared to word embedding modules that only perform embedding lookups.
  * The use of an accelerator is recommended.
  *
  * '''word_emb''': the character-based word representations with shape [batch_size, max_length, 512].  == word_emb
  *
  * '''lstm_outputs1''': the first LSTM hidden state with shape [batch_size, max_length, 1024]. === lstm_outputs1
  *
  * '''lstm_outputs2''': the second LSTM hidden state with shape [batch_size, max_length, 1024]. === lstm_outputs2
  *
  * '''elmo''': the weighted sum of the 3 layers, where the weights are trainable. This tensor has shape [batch_size, max_length, 1024]  == elmo
  *
  * See [[ https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/ElmoEmbeddingsTestSpec.scala ]] for further reference on how to use this API.
  *
  * '''Sources :'''
  *
  * [[https://tfhub.dev/google/elmo/3]]
  *
  * [[https://arxiv.org/abs/1802.05365]]
  *
  * ''' Paper abstract : '''
  *
  * We introduce a new type of deep contextualized word representation that models both (1) complex characteristics of word use (e.g., syntax and semantics), and (2) how these uses vary across linguistic contexts (i.e., to model polysemy). Our word vectors are learned functions of the internal states of a deep bidirectional language model (biLM), which is pre-trained on a large text corpus. We show that these representations can be easily added to existing models and significantly improve the state of the art across six challenging NLP problems, including question answering, textual entailment and sentiment analysis. We also present an analysis showing that exposing the deep internals of the pre-trained network is crucial, allowing downstream models to mix different types of semi-supervision signals.
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
  * @groupdesc Parameters A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  */
class ElmoEmbeddings(override val uid: String)
  extends AnnotatorModel[ElmoEmbeddings]
    with HasSimpleAnnotate[ElmoEmbeddings]
    with WriteTensorflowModel
    with HasEmbeddingsProperties
    with HasStorageRef
    with HasCaseSensitiveProperties {


  /** Output annotator type : TOKEN
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.DOCUMENT, AnnotatorType.TOKEN)
  /** Output annotator type : WORD_EMBEDDINGS
    *
    * @group anno
    **/
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.WORD_EMBEDDINGS
  /** Batch size. Large values allows faster processing but requires more memory.
    *
    * @group param
    **/
  val batchSize = new IntParam(this, "batchSize", "Batch size. Large values allows faster processing but requires more memory.")
  /** ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()
    *
    * @group param
    **/
  val configProtoBytes = new IntArrayParam(this, "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")
  /** Set ELMO pooling layer to: word_emb, lstm_outputs1, lstm_outputs2, or elmo
    *
    * @group param
    **/
  val poolingLayer = new Param[String](this, "poolingLayer", "Set ELMO pooling layer to: word_emb, lstm_outputs1, lstm_outputs2, or elmo")
  /** @group param */
  private var _model: Option[Broadcast[TensorflowElmo]] = None

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  def this() = this(Identifiable.randomUID("ELMO_EMBEDDINGS"))


  /** Large values allows faster processing but requires more memory.
    *
    * @group setParam
    **/
  def setBatchSize(size: Int): this.type = {
    if (get(batchSize).isEmpty)
      set(batchSize, size)
    this
  }

  /** Set Dimension of pooling layer. This is meta for the annotation and will not affect the actual embedding calculation.
    *
    * @group setParam
    **/
  override def setDimension(value: Int): this.type = {
    if (get(dimension).isEmpty)
      set(this.dimension, value)
    this

  }

  /** ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()
    *
    * @group setParam
    **/
  def setConfigProtoBytes(bytes: Array[Int]): ElmoEmbeddings.this.type = set(this.configProtoBytes, bytes)

  /** Function used to set the embedding output layer of the ELMO model
    * word_emb: the character-based word representations with shape [batch_size, max_length, 512].  == word_emb
    * lstm_outputs1: the first LSTM hidden state with shape [batch_size, max_length, 1024]. === lstm_outputs1
    * lstm_outputs2: the second LSTM hidden state with shape [batch_size, max_length, 1024]. === lstm_outputs2
    * elmo: the weighted sum of the 3 layers, where the weights are trainable. This tensor has shape [batch_size, max_length, 1024]  == elmo
    *
    * @param layer Layer specification
    * @group setParam
    */
  def setPoolingLayer(layer: String): this.type = {
    layer match {
      case "word_emb" => set(poolingLayer, "word_emb")
      case "lstm_outputs1" => set(poolingLayer, "lstm_outputs1")
      case "lstm_outputs2" => set(poolingLayer, "lstm_outputs2")
      case "elmo" => set(poolingLayer, "elmo")

      case _ => throw new MatchError("poolingLayer must be either word_emb, lstm_outputs1, lstm_outputs2, or elmo")
    }
  }


  /** Function used to set the embedding output layer of the ELMO model
    * word_emb: the character-based word representations with shape [batch_size, max_length, 512].  == word_emb
    * lstm_outputs1: the first LSTM hidden state with shape [batch_size, max_length, 1024]. === lstm_outputs1
    * lstm_outputs2: the second LSTM hidden state with shape [batch_size, max_length, 1024]. === lstm_outputs2
    * elmo: the weighted sum of the 3 layers, where the weights are trainable. This tensor has shape [batch_size, max_length, 1024]  == elmo
    *
    * @group getParam
    */
  def getPoolingLayer: String = $(poolingLayer)

  setDefault(
    batchSize -> 32,
    poolingLayer -> "word_emb",
    dimension -> 512
  )

  /** @group setParam */
  def setModelIfNotSet(spark: SparkSession, tensorflow: TensorflowWrapper): this.type = {
    if (_model.isEmpty) {

      _model = Some(
        spark.sparkContext.broadcast(
          new TensorflowElmo(
            tensorflow,
            batchSize = $(batchSize),
            configProtoBytes = getConfigProtoBytes
          )
        )
      )
    }

    this
  }

  /**
    * takes a document and annotations and produces new annotations of this annotator's annotation type
    *
    * @param annotations Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return any number of annotations processed for every input annotation. Not necessary one to one relationship
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences = TokenizedWithSentence.unpack(annotations)
    if (sentences.nonEmpty) {
      val embeddings = getModelIfNotSet.calculateEmbeddings(sentences, $(poolingLayer))

      WordpieceEmbeddingsSentence.pack(embeddings)
    } else {
      Seq.empty[Annotation]
    }
  }

  def getModelIfNotSet: TensorflowElmo = _model.get.value

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModel(path, spark, getModelIfNotSet.tensorflow, "_elmo", ElmoEmbeddings.tfFile, configProtoBytes = getConfigProtoBytes)
  }

  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  override protected def afterAnnotate(dataset: DataFrame): DataFrame = {
    dataset.withColumn(getOutputCol, wrapEmbeddingsMetadata(dataset.col(getOutputCol), $(dimension), Some($(storageRef))))
  }

}

trait ReadablePretrainedElmoModel extends ParamsAndFeaturesReadable[ElmoEmbeddings] with HasPretrained[ElmoEmbeddings] {
  override val defaultModelName: Some[String] = Some("elmo")

  /** Java compliant-overrides */
  override def pretrained(): ElmoEmbeddings = super.pretrained()
  override def pretrained(name: String): ElmoEmbeddings = super.pretrained(name)
  override def pretrained(name: String, lang: String): ElmoEmbeddings = super.pretrained(name, lang)
  override def pretrained(name: String, lang: String, remoteLoc: String): ElmoEmbeddings = super.pretrained(name, lang, remoteLoc)
}

trait ReadElmoTensorflowModel extends ReadTensorflowModel {
  this: ParamsAndFeaturesReadable[ElmoEmbeddings] =>

  override val tfFile: String = "elmo_tensorflow"

  def readTensorflow(instance: ElmoEmbeddings, path: String, spark: SparkSession): Unit = {
    val tf = readTensorflowModel(path, spark, "_elmo_tf", initAllTables = true)
    instance.setModelIfNotSet(spark, tf)
  }

  addReader(readTensorflow)

  def loadSavedModel(folder: String, spark: SparkSession): ElmoEmbeddings = {

    val f = new File(folder)
    val savedModel = new File(folder, "saved_model.pb")
    require(f.exists, s"Folder $folder not found")
    require(f.isDirectory, s"File $folder is not folder")
    require(
      savedModel.exists(),
      s"savedModel file saved_model.pb not found in folder $folder"
    )

    val wrapper = TensorflowWrapper.read(folder, zipped = false, useBundle = true, tags = Array("serve"), initAllTables = true)

    new ElmoEmbeddings()
      .setModelIfNotSet(spark, wrapper)
  }
}


object ElmoEmbeddings extends ReadablePretrainedElmoModel with ReadElmoTensorflowModel
