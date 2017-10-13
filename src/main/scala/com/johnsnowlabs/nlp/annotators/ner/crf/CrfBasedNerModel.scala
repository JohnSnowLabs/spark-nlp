package com.johnsnowlabs.nlp.annotators.ner.crf

import com.johnsnowlabs.ml.crf.{LinearChainCrfModel, SerializedLinearChainCrfModel}
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.common.{IndexedTaggedWord, TaggedSentence}
import com.johnsnowlabs.nlp.annotators.ner.crf.Annotated.{NerTaggedSentence, PosTaggedSentence}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.{Param, StringArrayParam}
import org.apache.spark.ml.util._
import org.apache.spark.sql.{Dataset, Encoders, Row, SQLContext}


/*
  Named Entity Recognition model
 */
class CrfBasedNerModel (override val uid: String)
  extends AnnotatorModel[CrfBasedNerModel] {

  def this() = this(Identifiable.randomUID("NER"))

  val entities = new StringArrayParam(this, "entities", "List of Entities to recognize")
  var model: Option[LinearChainCrfModel] = None
  var dictionaryFeatures = DictionaryFeatures(Seq.empty)

  def setModel(crf: LinearChainCrfModel): CrfBasedNerModel = {
    model = Some(crf)
    this
  }

  def setDictionaryFeatures(dictFeatures: DictionaryFeatures) = {
    dictionaryFeatures = dictFeatures
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
      val instance = FeatureGenerator(dictionaryFeatures).generate(sentence, crf.metadata)
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
      val crfModel = loaded.as[SerializedLinearChainCrfModel].head

      val dictPath = new Path(path, "dict").toString
      val dictLoaded = sparkSession.sqlContext.read.format("parquet")
        .load(dictPath)
        .collect
        .head

      val lines = dictLoaded.asInstanceOf[Row].getAs[Seq[String]](0)

      val dict = lines
        .map {line =>
          val items = line.split(":")
          (items(0), items(1))
        }
        .toMap

      val dictFeatures = new DictionaryFeatures(dict)

      instance
        .setModel(crfModel.deserialize)
        .setDictionaryFeatures(dictFeatures)
    }
  }

  class CrfBasedNerModelWriter(model: CrfBasedNerModel, baseWriter: MLWriter) extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      require(model.model.isDefined, "Crf Model must be defined before serialization")

      baseWriter.save(path)

      val spark = sparkSession
      import spark.sqlContext.implicits._

      val toStore = model.model.get.serialize
      val dataPath = new Path(path, "data").toString
      Seq(toStore).toDS.write.mode("overwrite").parquet(dataPath)

      val dictPath = new Path(path, "dict").toString
      val dictLines = model.dictionaryFeatures.dict.toSeq.map(p => p._1 + ":" + p._2)
      Seq(dictLines).toDS.write.mode("overwrite").parquet(dictPath)
    }
  }
}

