package com.johnsnowlabs.nlp.annotators.ner.crf

import com.johnsnowlabs.nlp._
import org.scalatest.{BeforeAndAfter, FlatSpec}
import com.johnsnowlabs.testutils.TravisCIHelper


class NerCrfApproachSpec extends FlatSpec {
  val spark = SparkAccessor.spark

  val nerSentence = DataBuilder.buildNerDataset(ContentProvider.nerCorpus)
  val nerModel = AnnotatorBuilder.getNerCrfModel(nerSentence)

  // Dataset ready for NER tagger
  lazy val nerInputDataset = AnnotatorBuilder.withFullPOSTagger(AnnotatorBuilder.withTokenizer(nerSentence))

  before {
    TravisCIHelper.startLogger
  }

  "NerCrfApproach" should "be serializable and deserializable correctly" in {
    nerModel.write.overwrite.save("./test_crf_pipeline")
    val loadedNer = NerCrfModel.read.load("./test_crf_pipeline")

    assert(nerModel.model.getOrDefault.serialize == loadedNer.model.getOrDefault.serialize)
    assert(nerModel.dictionaryFeatures.getOrDefault == loadedNer.dictionaryFeatures.getOrDefault)
  }


  it should "have correct set of labels" in {
    assert(nerModel.model.isSet)
    val metadata = nerModel.model.getOrDefault.metadata
    assert(metadata.labels.toSeq == Seq("@#Start", "PER", "O", "ORG", "LOC"))
  }


  it should "correctly store annotations" in {
    val tagged = nerModel.transform(nerInputDataset)
    val annotations = Annotation.collect(tagged, "ner").flatten.toSeq
    val labels = Annotation.collect(tagged, "label").flatten.toSeq

    assert(annotations.length == labels.length)
    for ((annotation, label) <- annotations.zip(labels)) {
      assert(annotation.start == label.start)
      assert(annotation.end == label.end)
      assert(annotation.annotatorType == AnnotatorType.NAMED_ENTITY)
      assert(annotation.result == label.result)
      assert(annotation.metadata.contains("word"))
    }
  }


  it should "correctly tag sentences" in {
    val tagged = nerModel.transform(nerInputDataset)
    val annotations = Annotation.collect(tagged, "ner").flatten

    val tags = annotations.map(a => a.result).toSeq
    assert(tags == Seq("PER", "PER", "O", "O", "ORG", "LOC", "O"))
  }


  "NerCrfModel" should "correctly train using dataset from file" in {
    val tagged = AnnotatorBuilder.withNerCrfTagger(nerInputDataset)
    val annotations = Annotation.collect(tagged, "ner").flatten

    val tags = annotations.map(a => a.result).toSeq
    assert(tags == Seq("PER", "PER", "O", "O", "ORG", "LOC", "O"))
  }


  it should "correctly handle entities param" in {
    val restrictedModel = new NerCrfModel()
      .setEntities(Array("PER", "LOC"))
      .setModel(nerModel.model.getOrDefault)
      .setOutputCol(nerModel.getOutputCol)
      .setInputCols(nerModel.getInputCols)

    val tagged = restrictedModel.transform(nerInputDataset)
    val annotations = Annotation.collect(tagged, "ner").flatten
    val tags = annotations.map(a => a.result).toSeq

    assert(tags == Seq("PER", "PER", "LOC"))
  }
}
