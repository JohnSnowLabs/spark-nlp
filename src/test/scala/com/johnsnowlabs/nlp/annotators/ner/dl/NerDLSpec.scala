package com.johnsnowlabs.nlp.annotators.ner.dl

import com.johnsnowlabs.nlp._
import com.johnsnowlabs.util.FileHelper
import org.scalatest.FlatSpec


class NerDLSpec extends FlatSpec {
  val spark = SparkAccessor.spark

  val nerSentence = DataBuilder.buildNerDataset(ContentProvider.nerCorpus)
  System.out.println(s"number of sentences in dataset ${nerSentence.count()}")

  // Dataset ready for NER tagger
  val nerInputDataset = AnnotatorBuilder.withTokenizer(nerSentence)
  System.out.println(s"number of sentences in dataset ${nerInputDataset.count()}")

  val nerModel = AnnotatorBuilder.getNerDLModel(nerSentence)


  "NerDLApproach" should "correctly annotate" in {
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

  "NerDLApproach" should "correctly tag sentences" in {
    val tagged = nerModel.transform(nerInputDataset)
    val annotations = Annotation.collect(tagged, "ner").flatten

    val tags = annotations.map(a => a.result).toSeq
    assert(tags.toList == Seq("PER", "PER", "O", "O", "ORG", "LOC", "O"))
  }

  "NerDLModel" should "correctly train using dataset from file" in {
    val tagged = AnnotatorBuilder.withNerDLTagger(nerInputDataset)
    val annotations = Annotation.collect(tagged, "ner").flatten

    val tags = annotations.map(a => a.result).toSeq
    assert(tags.toList == Seq("PER", "PER", "O", "O", "ORG", "LOC", "O"))
  }

  "NerDLApproach" should "be serializable and deserializable correctly" in {
    nerModel.write.overwrite.save("./test_ner_dl")
    val loadedNer = NerDLModel.read.load("./test_ner_dl")
    FileHelper.delete("./test_ner_dl")

    // Test that params of loaded model are the same
    assert(loadedNer.datasetParams.getOrDefault == nerModel.datasetParams.getOrDefault)

    // Test that loaded model do the same predictions
    val tokenized = AnnotatorBuilder.withTokenizer(nerInputDataset)
    val tagged = loadedNer.transform(tokenized)
    val annotations = Annotation.collect(tagged, "ner").flatten

    val tags = annotations.map(a => a.result).toSeq
    assert(tags.toList == Seq("PER", "PER", "O", "O", "ORG", "LOC", "O"))
  }

  "NerDLApproach" should "correct search for suitable graphs" in {
    val smallGraphFile = NerDLApproach.searchForSuitableGraph(10, 100, 100)
    assert(smallGraphFile.endsWith("char_cnn_blstm_10_100_100_25_30.pb"))

    val bigGraphFile = NerDLApproach.searchForSuitableGraph(25, 100, 100)
    assert(bigGraphFile.endsWith("char_cnn_blstm_30_100_100_25_30.pb"))

    assertThrows[IllegalArgumentException](NerDLApproach.searchForSuitableGraph(10, 101, 100))
    assertThrows[IllegalArgumentException](NerDLApproach.searchForSuitableGraph(10, 100, 101))
    assertThrows[IllegalArgumentException](NerDLApproach.searchForSuitableGraph(31, 100, 101))
  }
}

