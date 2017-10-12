package com.jsl.nlp.annotators.ner.crf

import com.jsl.nlp._
import org.scalatest.FlatSpec


class CrfBasedNerPipelineSpec extends FlatSpec {
  val nerSentence = DataBuilder.buildNerDataset(ContentProvider.nerCorpus)
  val nerModel = AnnotatorBuilder.getCrfBasedNerModel(nerSentence)

  // Dataset ready for NER tagger
  val nerInputDataset = AnnotatorBuilder.withFullPOSTagger(AnnotatorBuilder.withTokenizer(nerSentence))


  "CrfBasedNerModel" should "be serializable and deserializable correctly" in {
    nerModel.write.overwrite.save("./test_crf_pipeline")
    val loadedNer = CrfBasedNerModel.read.load("./test_crf_pipeline")

    assert(nerModel.model.get.serialize == loadedNer.model.get.serialize)
    assert(nerModel.dictionaryFeatures == loadedNer.dictionaryFeatures)
  }


  "CrfBasedNer" should "have correct set of labels" in {
    assert(nerModel.model.isDefined)
    val metadata = nerModel.model.get.metadata
    assert(metadata.labels.toSeq == Seq("@#Start", "PER", "O", "ORG", "LOC"))
  }


  "CrfBasedNer" should "correctly store annotations" in {
    val tagged = nerModel.transform(nerInputDataset)
    val annotations = Annotation.collect(tagged, "ner").flatten.toSeq
    val labels = Annotation.collect(tagged, "label").flatten.toSeq

    assert(annotations.length == labels.length)
    for ((annotation, label) <- annotations.zip(labels)) {
      assert(annotation.begin == label.begin)
      assert(annotation.end == label.end)
      assert(annotation.annotatorType == AnnotatorType.NAMED_ENTITY)
      assert(annotation.metadata("tag") == label.metadata("tag"))
      assert(annotation.metadata.contains("word"))
    }
  }

  "CrfBasedNer" should "correctly tag sentences" in {
    val tagged = nerModel.transform(nerInputDataset)
    val annotations = Annotation.collect(tagged, "ner").flatten

    val tags = annotations.map(a => a.metadata("tag")).toSeq
    assert(tags == Seq("PER", "PER", "O", "O", "ORG", "LOC", "O"))
  }


  "CrfBasedNerModel" should "correctly handle entities param" in {
    val restrictedModel = new CrfBasedNerModel()
      .setEntities(Array("PER", "LOC"))
      .setModel(nerModel.model.get)
      .setOutputCol(nerModel.getOutputCol)
      .setInputCols(nerModel.getInputCols)

    val tagged = restrictedModel.transform(nerInputDataset)
    val annotations = Annotation.collect(tagged, "ner").flatten
    val tags = annotations.map(a => a.metadata("tag")).toSeq

    assert(tags == Seq("PER", "PER", "LOC"))
  }

}
