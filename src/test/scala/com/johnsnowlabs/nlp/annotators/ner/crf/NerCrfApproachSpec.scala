package com.johnsnowlabs.nlp.annotators.ner.crf

import com.johnsnowlabs.nlp._
import org.scalatest.FlatSpec


class NerCrfApproachSpec extends FlatSpec {
  val nerSentence = DataBuilder.buildNerDataset(ContentProvider.nerCorpus)
  val nerModel = AnnotatorBuilder.getNerCrfModel(nerSentence)

  // Dataset ready for NER tagger
  val nerInputDataset = AnnotatorBuilder.withFullPOSTagger(AnnotatorBuilder.withTokenizer(nerSentence))


  "NerCrfApproach" should "be serializable and deserializable correctly" in {
    nerModel.write.overwrite.save("./test_crf_pipeline")
    val loadedNer = NerCrfModel.read.load("./test_crf_pipeline")

    assert(nerModel.model.getValue.serialize == loadedNer.model.getValue.serialize)
    assert(nerModel.dictionaryFeatures.getValue == loadedNer.dictionaryFeatures.getValue)
  }


  "NerCrfApproach" should "have correct set of labels" in {
    assert(nerModel.model.isSet)
    val metadata = nerModel.model.getValue.metadata
    assert(metadata.labels.toSeq == Seq("@#Start", "PER", "O", "ORG", "LOC"))
  }


  "NerCrfApproach" should "correctly store annotations" in {
    val tagged = nerModel.transform(nerInputDataset)
    val annotations = Annotation.collect(tagged, "ner").flatten.toSeq
    val labels = Annotation.collect(tagged, "label").flatten.toSeq

    assert(annotations.length == labels.length)
    for ((annotation, label) <- annotations.zip(labels)) {
      assert(annotation.begin == label.begin)
      assert(annotation.end == label.end)
      assert(annotation.annotatorType == AnnotatorType.NAMED_ENTITY)
      assert(annotation.result == label.result)
      assert(annotation.metadata.contains("word"))
    }
  }


  "NerCrfApproach" should "correctly tag sentences" in {
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


  "NerCrfModel" should "correctly handle entities param" in {
    val restrictedModel = new NerCrfModel()
      .setEntities(Array("PER", "LOC"))
      .setModel(nerModel.model.getValue)
      .setOutputCol(nerModel.getOutputCol)
      .setInputCols(nerModel.getInputCols)

    val tagged = restrictedModel.transform(nerInputDataset)
    val annotations = Annotation.collect(tagged, "ner").flatten
    val tags = annotations.map(a => a.result).toSeq

    assert(tags == Seq("PER", "PER", "LOC"))
  }
}
