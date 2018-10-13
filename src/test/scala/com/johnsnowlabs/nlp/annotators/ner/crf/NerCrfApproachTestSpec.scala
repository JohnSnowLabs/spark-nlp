package com.johnsnowlabs.nlp.annotators.ner.crf

import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.embeddings.EmbeddingsHelper
import org.scalatest.FlatSpec

class NerCrfApproachTestSpec extends FlatSpec {
  val spark = SparkAccessor.spark

  val nerSentence = DataBuilder.buildNerDataset(ContentProvider.nerCorpus)
  System.out.println(s"number of sentences in dataset ${nerSentence.count()}")

  // Dataset ready for NER tagger
  val nerInputDataset = AnnotatorBuilder.withFullPOSTagger(nerSentence)
  System.out.println(s"number of sentences in dataset ${nerInputDataset.count()}")
  val nerModel = AnnotatorBuilder.getNerCrfModel(nerSentence)

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
      assert(annotation.begin == label.begin)
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
    assert(tags.toList == Seq("PER", "PER", "O", "O", "ORG", "LOC", "O"))
  }


  "NerCrfModel" should "correctly train using dataset from file" in {
    val tagged = AnnotatorBuilder.withNerCrfTagger(nerInputDataset)
    val annotations = Annotation.collect(tagged, "ner").flatten

    val tags = annotations.map(a => a.result).toSeq
    assert(tags.toList == Seq("PER", "PER", "O", "O", "ORG", "LOC", "O"))
  }


  it should "correctly handle entities param" in {

    val smallEmbeddings = EmbeddingsHelper.loadEmbeddings(
      "src/test/resources/random_embeddings_dim4.txt",
      spark,
      "TEXT",
      4,
      false
    )

    EmbeddingsHelper.setEmbeddingsRef("random_embeddings", smallEmbeddings)

    val restrictedModel = new NerCrfModel()
      .setEntities(Array("PER", "LOC"))
      .setModel(nerModel.model.getOrDefault)
      .setOutputCol(nerModel.getOutputCol)
      .setInputCols(nerModel.getInputCols)
      .setEmbeddingsDim(smallEmbeddings.dim)
      .setEmbeddingsRef("random_embeddings")
      .setCaseSensitiveEmbeddings(smallEmbeddings.caseSensitive)

    val tagged = restrictedModel.transform(nerInputDataset)
    val annotations = Annotation.collect(tagged, "ner").flatten
    val tags = annotations.map(a => a.result).toSeq

    assert(tags == Seq("PER", "PER", "LOC"))
  }

}
