package com.johnsnowlabs.nlp.annotators.ner.crf

import com.johnsnowlabs.ml.crf.{LinearChainCrfModel, SerializedLinearChainCrfModel}
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.common.{IndexedTaggedWord, NerTagged, PosTagged, TaggedSentence}
import com.johnsnowlabs.nlp.annotators.common.Annotated.{NerTaggedSentence, PosTaggedSentence}
import com.johnsnowlabs.nlp.embeddings.ModelWithWordEmbeddings
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.StringArrayParam
import org.apache.spark.ml.util._
import org.apache.spark.sql.{Encoders, Row}


/*
  Named Entity Recognition model
 */
class NerCrfModel(override val uid: String) extends AnnotatorModel[NerCrfModel]
with ModelWithWordEmbeddings[NerCrfModel]{

  def this() = this(Identifiable.randomUID("NER"))

  val entities = new StringArrayParam(this, "entities", "List of Entities to recognize")
  var model: Option[LinearChainCrfModel] = None
  var dictionaryFeatures = DictionaryFeatures(Seq.empty)

  def setModel(crf: LinearChainCrfModel): NerCrfModel = {
    model = Some(crf)
    this
  }

  def setDictionaryFeatures(dictFeatures: DictionaryFeatures) = {
    dictionaryFeatures = dictFeatures
    this
  }

  def setEntities(toExtract: Array[String]): NerCrfModel = set(entities, toExtract)

  /**
  Predicts Named Entities in input sentences
    * @param sentences POS tagged sentences.
    * @return sentences with recognized Named Entities
    */
  def tag(sentences: Seq[PosTaggedSentence]): Seq[NerTaggedSentence] = {
    require(model.isDefined, "model must be set before tagging")

    val crf = model.get

    val fg = FeatureGenerator(dictionaryFeatures, embeddings)
    sentences.map{sentence =>
      val instance = fg.generate(sentence, crf.metadata)
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

  def shrink(minW: Float): NerCrfModel = {
    model = model.map(m => m.shrink(minW))
    this
  }

  override val requiredAnnotatorTypes = Array(DOCUMENT, TOKEN, POS)

  override val annotatorType: AnnotatorType = NAMED_ENTITY

  override def write: MLWriter = new NerCrfModel.NerCrfModelWriter(this, super.write)
}

object NerCrfModel extends DefaultParamsReadable[NerCrfModel] {
  implicit val crfEncoder = Encoders.kryo[SerializedLinearChainCrfModel]

  override def read: MLReader[NerCrfModel] = new NerCrfModelReader(super.read)

  class NerCrfModelReader(baseReader: MLReader[NerCrfModel]) extends MLReader[NerCrfModel] {
    override def load(path: String): NerCrfModel = {
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

      instance.deserializeEmbeddings(path, sparkSession.sparkContext)
      instance
    }
  }

  class NerCrfModelWriter(model: NerCrfModel, baseWriter: MLWriter) extends MLWriter {

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

      model.serializeEmbeddings(path, sparkSession.sparkContext)
    }
  }
}
