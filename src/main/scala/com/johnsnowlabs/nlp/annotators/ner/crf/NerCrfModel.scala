package com.johnsnowlabs.nlp.annotators.ner.crf

import com.johnsnowlabs.ml.crf.LinearChainCrfModel
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.common.Annotated.{NerTaggedSentence, PosTaggedSentence}
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.johnsnowlabs.nlp.serialization.{MapFeature, StructFeature}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, ParamsAndFeaturesReadable}
import org.apache.spark.ml.param.StringArrayParam
import org.apache.spark.ml.util._


/*
  Named Entity Recognition model
 */

class NerCrfModel(override val uid: String) extends AnnotatorModel[NerCrfModel] {

  def this() = this(Identifiable.randomUID("NER"))

  val entities = new StringArrayParam(this, "entities", "List of Entities to recognize")
  val model: StructFeature[LinearChainCrfModel] = new StructFeature[LinearChainCrfModel](this, "crfModel")
  val dictionaryFeatures: MapFeature[String, String] = new MapFeature[String, String](this, "dictionaryFeatures")

  def setModel(crf: LinearChainCrfModel): NerCrfModel = set(model, crf)

  def setDictionaryFeatures(dictFeatures: DictionaryFeatures): this.type = set(dictionaryFeatures, dictFeatures.dict)
  setDefault(dictionaryFeatures, () => Map.empty[String, String])

  def setEntities(toExtract: Array[String]): NerCrfModel = set(entities, toExtract)

  /**
  Predicts Named Entities in input sentences
    * @param sentences POS tagged sentences.
    * @return sentences with recognized Named Entities
    */
  def tag(sentences: Seq[(PosTaggedSentence, WordpieceEmbeddingsSentence)]): Seq[NerTaggedSentence] = {
    require(model.isSet, "model must be set before tagging")

    val crf = $$(model)

    val fg = FeatureGenerator(new DictionaryFeatures($$(dictionaryFeatures)))
    sentences.map{case (sentence, withEmbeddings) =>
      val instance = fg.generate(sentence, withEmbeddings, crf.metadata)
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

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sourceSentences = PosTagged.unpack(annotations)
    val withEmbeddings = WordpieceEmbeddingsSentence.unpack(annotations)
    val taggedSentences = tag(sourceSentences.zip(withEmbeddings))
    NerTagged.pack(taggedSentences)
  }

  def shrink(minW: Float): NerCrfModel = set(model, $$(model).shrink(minW))

  override val requiredAnnotatorTypes = Array(DOCUMENT, TOKEN, POS, WORD_EMBEDDINGS)

  override val annotatorType: AnnotatorType = NAMED_ENTITY

}

trait PretrainedNerCrf {
  def pretrained(name: String = "ner_fast", language: Option[String] = Some("en"), remoteLoc: String = ResourceDownloader.publicLoc): NerCrfModel =
    ResourceDownloader.downloadModel(NerCrfModel, name, language, remoteLoc)
}

object NerCrfModel extends ParamsAndFeaturesReadable[NerCrfModel] with PretrainedNerCrf
