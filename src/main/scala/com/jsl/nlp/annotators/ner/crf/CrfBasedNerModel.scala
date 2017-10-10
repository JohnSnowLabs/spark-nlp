package com.jsl.nlp.annotators.ner.crf

import com.jsl.ml.crf.VectorMath.Vector
import com.jsl.ml.crf.{DatasetMetadata, LinearChainCrfModel, SerializedLinearChainCrfModel}
import com.jsl.nlp.AnnotatorType._
import com.jsl.nlp.annotators.common.{IndexedTaggedWord, TaggedSentence}
import com.jsl.nlp.annotators.ner.crf.Annotated.{NerTaggedSentence, PosTaggedSentence}
import com.jsl.nlp.{Annotation, AnnotatorModel}
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util._
import org.apache.spark.sql.Encoders


/*
  Named Entity Recognition model
 */
class CrfBasedNerModel (override val uid: String)
  extends AnnotatorModel[CrfBasedNerModel] {

  def this() = this(Identifiable.randomUID("NER"))

  val entities = new Param[Array[String]](this, "entities", "List of Entities to recognize")
  var model: Option[LinearChainCrfModel] = None

  def setModel(crf: LinearChainCrfModel): CrfBasedNerModel = {
    model = Some(crf)
    this
  }

  def setEntities(toExtract: Array[String]): CrfBasedNerModel = set(entities, toExtract)

  /**
    Predicts Named Entities in input sentences
    * @param sentences POS tagged sentences.
    * @return sentences with recognized Named Entities
    */
  def tag(sentences: Seq[PosTaggedSentence]): Seq[NerTaggedSentence] = {
    require(model.isDefined, "model must be set before tagging")

    val crf = model.get

    sentences.map{sentence =>
      val instance = FeatureGenerator.generate(sentence, crf.metadata)
      val labelIds = crf.predict(instance)
      val words = sentence.indexedTaggedWords
        .zip(labelIds.labels)
        .flatMap{case (word, labelId) =>
          val label = crf.metadata.labels(labelId)

          if (!isDefined(entities) || $(entities).isEmpty || $(entities).contains(label)) {
            Some(IndexedTaggedWord(word.word, label, word.begin, word.end))
          }
          else {
            None
          }
        }

      TaggedSentence(words)
    }
  }

  override protected def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sourceSentences = PosTagged.unpack(annotations)
    val taggedSentences = tag(sourceSentences)
    NerTagged.pack(taggedSentences)
  }

  def shrink(minW: Float): CrfBasedNerModel = {
    model = model.map(m => m.shrink(minW))
    this
  }

  override val requiredAnnotatorTypes = Array(DOCUMENT, TOKEN, POS)

  override val annotatorType: AnnotatorType = NAMED_ENTITY

  override def write: MLWriter = new CrfBasedNerModel.CrfBasedNerModelWriter(this, super.write)
}

object CrfBasedNerModel extends DefaultParamsReadable[CrfBasedNerModel] {
  implicit val crfEncoder = Encoders.kryo[SerializedLinearChainCrfModel]

  override def read: MLReader[CrfBasedNerModel] = new CrfBasedNerModelReader(super.read)

  class CrfBasedNerModelReader(baseReader: MLReader[CrfBasedNerModel]) extends MLReader[CrfBasedNerModel] {
    override def load(path: String): CrfBasedNerModel = {
      val instance = baseReader.load(path)

      val dataPath = new Path(path, "data").toString

      val loaded = sparkSession.sqlContext.read.format("parquet").load(dataPath)
      val loadedDs = loaded.as[SerializedLinearChainCrfModel]
      val crfModel = loadedDs.head

      instance.setModel(crfModel.deserialize)
    }
  }

  class CrfBasedNerModelWriter(model: CrfBasedNerModel, baseWriter: MLWriter) extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      baseWriter.save(path)

      require(model.model.isDefined, "Crf Model must be defined before serialization")

      val toStore = model.model.get.serialize
      val dataPath = new Path(path, "data").toString

      val df = sparkSession.sparkContext.parallelize(Seq(toStore)).repartition(1)
      val ds = sparkSession.sqlContext.implicits.rddToDatasetHolder(df).toDS
      ds.write.mode("overwrite").parquet(dataPath)
    }
  }
}

