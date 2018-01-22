package com.johnsnowlabs.nlp.annotators.ner.crf

import com.johnsnowlabs.ml.crf.{CrfParams, LinearChainCrf, TextSentenceLabels, Verbose}
import com.johnsnowlabs.nlp.{AnnotatorApproach, AnnotatorType, DocumentAssembler}
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, NAMED_ENTITY, POS, TOKEN}
import com.johnsnowlabs.nlp.annotators.RegexTokenizer
import com.johnsnowlabs.nlp.annotators.common.Annotated.PosTaggedSentence
import com.johnsnowlabs.nlp.annotators.common.NerTagged
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronApproach
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetectorModel
import com.johnsnowlabs.nlp.datasets.CoNLL
import com.johnsnowlabs.nlp.embeddings.AnnotatorWithWordEmbeddings
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.param.{DoubleParam, IntParam, Param, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}

/*
  Algorithm for training Named Entity Recognition Model.
   */
class NerCrfApproach(override val uid: String) extends AnnotatorApproach[NerCrfModel]
  with AnnotatorWithWordEmbeddings[NerCrfApproach, NerCrfModel] {

  def this() = this(Identifiable.randomUID("NER"))

  override val description = "CRF based Named Entity Recognition Tagger"
  override val requiredAnnotatorTypes = Array(DOCUMENT, TOKEN, POS)
  override val annotatorType = NAMED_ENTITY

  val labelColumn = new Param[String](this, "labelColumn", "Column with label per each token")
  val entities = new StringArrayParam(this, "entities", "Entities to recognize")

  val minEpochs = new IntParam(this, "minEpochs", "Minimum number of epochs to train")
  val maxEpochs = new IntParam(this, "maxEpochs", "Maximum number of epochs to train")
  val l2 = new DoubleParam(this, "l2", "L2 regularization coefficient")
  val c0 = new IntParam(this, "c0", "c0 params defining decay speed for gradient")
  val lossEps = new DoubleParam(this, "lossEps", "If Epoch relative improvement less than eps then training is stopped")
  val minW = new DoubleParam(this, "minW", "Features with less weights then this param value will be filtered")

  val dicts = new StringArrayParam(this, "dicts", "Additional dictionary paths to use as a features")

  val verbose = new IntParam(this, "verbose", "Level of verbosity during training")
  val randomSeed = new IntParam(this, "randomSeed", "Random seed")

  val datasetPath = new Param[String](this, "datasetPath", "Path to dataset. " +
    "If path is empty will use dataset passed to train as usual Spark Pipeline stage")

  def setLabelColumn(column: String) = set(labelColumn, column)
  def setEntities(tags: Array[String]) = set(entities, tags)

  def setMinEpochs(epochs: Int) = set(minEpochs, epochs)
  def setMaxEpochs(epochs: Int) = set(maxEpochs, epochs)
  def setL2(l2: Double) = set(this.l2, l2)
  def setC0(c0: Int) = set(this.c0, c0)
  def setLossEps(eps: Double) = set(this.lossEps, eps)
  def setMinW(w: Double) = set(this.minW, w)

  def setDicts(paths: Seq[String]) = set(dicts, paths.toArray)

  def setVerbose(verbose: Int) = set(this.verbose, verbose)
  def setVerbose(verbose: Verbose.Level) = set(this.verbose, verbose.id)
  def setRandomSeed(seed: Int) = set(randomSeed, seed)

  def setDatsetPath(path: String) = set(datasetPath, path)

  setDefault(
    minEpochs -> 0,
    maxEpochs -> 1000,
    l2 -> 1f,
    c0 -> 2250000,
    lossEps -> 1e-3f,
    verbose -> Verbose.Silent.id
  )


  private def getTrainDataframe(dataset: Dataset[_]): DataFrame = {

    if (!isDefined(datasetPath))
      return dataset.toDF()

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetectorModel()
      .setCustomBoundChars(Array("\n\n"))
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer = new RegexTokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val posTagger = new PerceptronApproach()
      .setCorpusPath("anc-pos-corpus/")
      .setNIterations(10)
      .setInputCols("token", "document")
      .setOutputCol("pos")

    val pipeline = new Pipeline().setStages(
      Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger)
    )

    val reader = CoNLL(3, AnnotatorType.NAMED_ENTITY)
    val dataframe = reader.readDataset($(datasetPath), dataset.sparkSession).toDF
    pipeline.fit(dataframe).transform(dataframe)
  }


  override def train(dataset: Dataset[_]): NerCrfModel = {

    val rows = getTrainDataframe(dataset)

    val trainDataset: Array[(TextSentenceLabels, PosTaggedSentence)] = NerTagged.collectTrainingInstances(rows, getInputCols, $(labelColumn))

    val dictPaths = get(dicts).getOrElse(Array.empty[String])
    val dictFeatures = DictionaryFeatures.read(dictPaths.toSeq)
    val crfDataset = FeatureGenerator(dictFeatures, embeddings)
      .generateDataset(trainDataset)

    val params = CrfParams(
      minEpochs = getOrDefault(minEpochs),
      maxEpochs = getOrDefault(maxEpochs),

      l2 = getOrDefault(l2).toFloat,
      c0 = getOrDefault(c0),
      lossEps = getOrDefault(lossEps).toFloat,

      verbose = Verbose.Epochs,
      randomSeed = get(randomSeed)
    )

    val crf = new LinearChainCrf(params)
    val crfModel = crf.trainSGD(crfDataset)

    var model = new NerCrfModel()
      .setModel(crfModel)
      .setDictionaryFeatures(dictFeatures)

    if (isDefined(entities))
      model.setEntities($(entities))

    if (isDefined(minW))
      model = model.shrink($(minW).toFloat)

    model
  }
}

object NerCrfApproach extends DefaultParamsReadable[NerCrfApproach]