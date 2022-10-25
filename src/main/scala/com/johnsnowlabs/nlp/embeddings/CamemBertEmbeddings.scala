package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.ml.tensorflow.sentencepiece._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}

import java.io.File

/** The CamemBERT model was proposed in CamemBERT: a Tasty French Language Model by Louis Martin,
  * Benjamin Muller, Pedro Javier Ortiz Suárez, Yoann Dupont, Laurent Romary, Éric Villemonte de
  * la Clergerie, Djamé Seddah, and Benoît Sagot. It is based on Facebook’s RoBERTa model released
  * in 2019. It is a model trained on 138GB of French text.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val embeddings = CamemBertEmbeddings.pretrained()
  *   .setInputCols("token", "document")
  *   .setOutputCol("camembert_embeddings")
  * }}}
  * The default model is `"camembert_base"`, if no name is provided.
  *
  * For available pretrained models please see the
  * [[https://nlp.johnsnowlabs.com/models?task=Embeddings Models Hub]].
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/blogposts/3.NER_with_BERT.ipynb Spark NLP Workshop]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/CamemBertEmbeddingsTestSpec.scala CamemBertEmbeddingsTestSpec]].
  * To see which models are compatible and how to import them see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]].
  *
  * '''Sources''' :
  *
  * [[https://arxiv.org/abs/1911.03894 CamemBERT: a Tasty French Language Model]]
  *
  * [[https://huggingface.co/camembert]]
  *
  * ''' Paper abstract '''
  *
  * ''Pretrained language models are now ubiquitous in Natural Language Processing. Despite their
  * success, most available models have either been trained on English data or on the
  * concatenation of data in multiple languages. This makes practical use of such models --in all
  * languages except English-- very limited. In this paper, we investigate the feasibility of
  * training monolingual Transformer-based language models for other languages, taking French as
  * an example and evaluating our language models on part-of-speech tagging, dependency parsing,
  * named entity recognition and natural language inference tasks. We show that the use of web
  * crawled data is preferable to the use of Wikipedia data. More surprisingly, we show that a
  * relatively small web crawled dataset (4GB) leads to results that are as good as those obtained
  * using larger datasets (130+GB). Our best performing model CamemBERT reaches or improves the
  * state of the art in all four downstream tasks.''
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.embeddings.CamemBertEmbeddings
  * import com.johnsnowlabs.nlp.EmbeddingsFinisher
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols("document")
  *   .setOutputCol("token")
  *
  * val embeddings = CamemBertEmbeddings.pretrained()
  *   .setInputCols("token", "document")
  *   .setOutputCol("camembert_embeddings")
  *
  * val embeddingsFinisher = new EmbeddingsFinisher()
  *   .setInputCols("camembert_embeddings")
  *   .setOutputCols("finished_embeddings")
  *   .setOutputAsVector(true)
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   tokenizer,
  *   embeddings,
  *   embeddingsFinisher
  * ))
  *
  * val data = Seq("C'est une phrase.").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
  * +--------------------------------------------------------------------------------+
  * |                                                                          result|
  * +--------------------------------------------------------------------------------+
  * |[0.08442357927560806,-0.12863239645957947,-0.03835778683423996,0.200479581952...|
  * |[0.048462312668561935,0.12637358903884888,-0.27429091930389404,-0.07516729831...|
  * |[0.02690504491329193,0.12104076147079468,0.012526623904705048,-0.031543646007...|
  * |[0.05877285450696945,-0.08773420006036758,-0.06381352990865707,0.122621834278...|
  * +--------------------------------------------------------------------------------+
  * }}}
  *
  * @see
  *   [[https://nlp.johnsnowlabs.com/docs/en/annotators Annotators Main Page]] for a list of
  *   transformer based embeddings
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
class CamemBertEmbeddings(override val uid: String)
    extends AnnotatorModel[CamemBertEmbeddings]
    with HasBatchedAnnotate[CamemBertEmbeddings]
    with WriteTensorflowModel
    with WriteSentencePieceModel
    with HasEmbeddingsProperties
    with HasStorageRef
    with HasCaseSensitiveProperties {

  def this() = this(Identifiable.randomUID("CAMEMBERT_EMBEDDINGS"))

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * `config_proto.SerializeToString()`
    *
    * @group param
    */
  val configProtoBytes = new IntArrayParam(
    this,
    "configProtoBytes",
    "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** @group setParam */
  def setConfigProtoBytes(bytes: Array[Int]): CamemBertEmbeddings.this.type =
    set(this.configProtoBytes, bytes)

  /** @group getParam */
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  /** Max sentence length to process (Default: `128`)
    *
    * @group param
    */
  val maxSentenceLength =
    new IntParam(this, "maxSentenceLength", "Max sentence length to process")

  /** @group setParam */
  def setMaxSentenceLength(value: Int): this.type = {
    require(
      value <= 512,
      "CamemBERT models do not support sequences longer than 512 because of trainable positional embeddings.")
    require(value >= 1, "The maxSentenceLength must be at least 1")
    set(maxSentenceLength, value)
    this
  }

  /** @group getParam */
  def getMaxSentenceLength: Int = $(maxSentenceLength)

  /** It contains TF model signatures for the laded saved model
    *
    * @group param
    */
  val signatures = new MapFeature[String, String](model = this, name = "signatures")

  /** @group setParam */
  def setSignatures(value: Map[String, String]): this.type = {
    if (get(signatures).isEmpty)
      set(signatures, value)
    this
  }

  /** @group getParam */
  def getSignatures: Option[Map[String, String]] = get(this.signatures)

  private var _model: Option[Broadcast[TensorflowCamemBert]] = None

  def setModelIfNotSet(
      spark: SparkSession,
      tensorflowWrapper: TensorflowWrapper,
      spp: SentencePieceWrapper): CamemBertEmbeddings = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new TensorflowCamemBert(
            tensorflowWrapper,
            spp,
            configProtoBytes = getConfigProtoBytes,
            signatures = getSignatures)))
    }

    this
  }

  /** @group getParam */
  def getModelIfNotSet: TensorflowCamemBert = _model.get.value

  /** Set Embeddings dimensions for the CamemBERT model Only possible to set this when the first
    * time is saved dimension is not changeable, it comes from CamemBERT config file
    *
    * @group setParam
    */
  override def setDimension(value: Int): this.type = {
    if (get(dimension).isEmpty)
      set(this.dimension, value)
    this
  }

  /** Whether to lowercase tokens or not
    *
    * @group setParam
    */
  override def setCaseSensitive(value: Boolean): this.type = {
    if (get(caseSensitive).isEmpty)
      set(this.caseSensitive, value)
    this
  }

  setDefault(batchSize -> 8, dimension -> 768, maxSentenceLength -> 128, caseSensitive -> true)

  /** takes a document and annotations and produces new annotations of this annotator's annotation
    * type
    *
    * @param batchedAnnotations
    *   Annotations in batches that correspond to inputAnnotationCols generated by previous
    *   annotators if any
    * @return
    *   any number of annotations processed for every batch of input annotations. Not necessary
    *   one to one relationship
    */
  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {

    // Unpack annotations and zip each sentence to the index or the row it belongs to
    val sentencesWithRow = batchedAnnotations.zipWithIndex
      .flatMap { case (annotations, i) =>
        TokenizedWithSentence.unpack(annotations).toArray.map(x => (x, i))
      }

    // Process all sentences
    val sentenceWordEmbeddings = getModelIfNotSet.predict(
      sentencesWithRow.map(_._1),
      $(batchSize),
      $(maxSentenceLength),
      $(caseSensitive))

    // Group resulting annotations by rows. If there are not sentences in a given row, return empty sequence
    batchedAnnotations.indices.map(rowIndex => {
      val rowEmbeddings = sentenceWordEmbeddings
        // zip each annotation with its corresponding row index
        .zip(sentencesWithRow)
        // select the sentences belonging to the current row
        .filter(_._2._2 == rowIndex)
        // leave the annotation only
        .map(_._1)

      if (rowEmbeddings.nonEmpty)
        WordpieceEmbeddingsSentence.pack(rowEmbeddings)
      else
        Seq.empty[Annotation]
    })

  }

  override protected def afterAnnotate(dataset: DataFrame): DataFrame = {
    dataset.withColumn(
      getOutputCol,
      wrapEmbeddingsMetadata(dataset.col(getOutputCol), $(dimension), Some($(storageRef))))
  }

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  override val inputAnnotatorTypes: Array[String] =
    Array(AnnotatorType.DOCUMENT, AnnotatorType.TOKEN)
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.WORD_EMBEDDINGS

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)

    writeTensorflowModelV2(
      path,
      spark,
      getModelIfNotSet.tensorflow,
      "_camembert",
      CamemBertEmbeddings.tfFile,
      configProtoBytes = getConfigProtoBytes)

    writeSentencePieceModel(
      path,
      spark,
      getModelIfNotSet.spp,
      "_camembert",
      CamemBertEmbeddings.sppFile)
  }

}

trait ReadablePretrainedCamemBertModel
    extends ParamsAndFeaturesReadable[CamemBertEmbeddings]
    with HasPretrained[CamemBertEmbeddings] {
  override val defaultModelName: Some[String] = Some("camembert_base")
  override val defaultLang: String = "fr"

  /** Java compliant-overrides */
  override def pretrained(): CamemBertEmbeddings = super.pretrained()

  override def pretrained(name: String): CamemBertEmbeddings = super.pretrained(name)

  override def pretrained(name: String, lang: String): CamemBertEmbeddings =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): CamemBertEmbeddings =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadCamemBertTensorflowModel extends ReadTensorflowModel with ReadSentencePieceModel {
  this: ParamsAndFeaturesReadable[CamemBertEmbeddings] =>

  override val tfFile: String = "camembert_tensorflow"
  override val sppFile: String = "camembert_spp"

  def readTensorflow(instance: CamemBertEmbeddings, path: String, spark: SparkSession): Unit = {

    val tf = readTensorflowModel(path, spark, "_camembert_tf", initAllTables = false)
    val spp = readSentencePieceModel(path, spark, "_camembert_spp", sppFile)
    instance.setModelIfNotSet(spark, tf, spp)
  }

  addReader(readTensorflow)

  def loadSavedModel(tfModelPath: String, spark: SparkSession): CamemBertEmbeddings = {

    val f = new File(tfModelPath)
    val savedModel = new File(tfModelPath, "saved_model.pb")
    require(f.exists, s"Folder $tfModelPath not found")
    require(f.isDirectory, s"File $tfModelPath is not folder")
    require(
      savedModel.exists(),
      s"savedModel file saved_model.pb not found in folder $tfModelPath")
    val sppModelPath = tfModelPath + "/assets"
    val sppModel = new File(sppModelPath, "sentencepiece.bpe.model")
    require(
      sppModel.exists(),
      s"SentencePiece model sentencepiece.bpe.model not found in folder $sppModelPath")

    val (wrapper, signatures) =
      TensorflowWrapper.read(tfModelPath, zipped = false, useBundle = true)
    val spp = SentencePieceWrapper.read(sppModel.toString)

    val _signatures = signatures match {
      case Some(s) => s
      case None => throw new Exception("Cannot load signature definitions from model!")
    }

    /** the order of setSignatures is important if we use getSignatures inside setModelIfNotSet */
    new CamemBertEmbeddings()
      .setSignatures(_signatures)
      .setModelIfNotSet(spark, wrapper, spp)
  }
}

/** This is the companion object of [[CamemBertEmbeddings]]. Please refer to that class for the
  * documentation.
  */
object CamemBertEmbeddings
    extends ReadablePretrainedCamemBertModel
    with ReadCamemBertTensorflowModel
