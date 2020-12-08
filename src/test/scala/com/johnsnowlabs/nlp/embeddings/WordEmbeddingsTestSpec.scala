package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.Chunk2Doc
import com.johnsnowlabs.nlp.annotator.{ChunkTokenizer, NerConverter, NerDLModel, SentenceDetectorDLModel}
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.base.{DocumentAssembler, RecursivePipeline}
import com.johnsnowlabs.nlp.util.io.{ReadAs, ResourceHelper}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.scalatest._

class WordEmbeddingsTestSpec extends FlatSpec {

  "Word Embeddings" should "correctly embed clinical words not embed non-existent words" ignore {

    val words = ResourceHelper.spark.read.option("header","true").csv("src/test/resources/embeddings/clinical_words.txt")
    val notWords = ResourceHelper.spark.read.option("header","true").csv("src/test/resources/embeddings/not_words.txt")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("word")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = WordEmbeddingsModel.pretrained()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)

    val pipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        tokenizer,
        embeddings
      ))

    val wordsP = pipeline.fit(words).transform(words).cache()
    val notWordsP = pipeline.fit(notWords).transform(notWords).cache()

    val wordsCoverage = WordEmbeddingsModel.withCoverageColumn(wordsP, "embeddings", "cov_embeddings")
    val notWordsCoverage = WordEmbeddingsModel.withCoverageColumn(notWordsP, "embeddings", "cov_embeddings")

    wordsCoverage.select("word","cov_embeddings").show(1)
    notWordsCoverage.select("word","cov_embeddings").show(1)

    val wordsOverallCoverage = WordEmbeddingsModel.overallCoverage(wordsCoverage,"embeddings").percentage
    val notWordsOverallCoverage = WordEmbeddingsModel.overallCoverage(notWordsCoverage,"embeddings").percentage

    ResourceHelper.spark.createDataFrame(
      Seq(
        ("Words", wordsOverallCoverage),("Not Words", notWordsOverallCoverage)
      )
    ).toDF("Dataset", "OverallCoverage").show(1)

    assert(wordsOverallCoverage == 1)
    assert(notWordsOverallCoverage == 0)
  }

  "Word Embeddings" should "store and load from disk" in {

    val data =
      ResourceHelper.spark.read.option("header","true").csv("src/test/resources/embeddings/clinical_words.txt")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("word")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = new WordEmbeddings()
      .setStoragePath("src/test/resources/random_embeddings_dim4.txt", ReadAs.TEXT)
      .setDimension(4)
      .setStorageRef("glove_4d")
      .setInputCols("document", "token")
      .setOutputCol("embeddings")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        tokenizer,
        embeddings
      ))

    val model = pipeline.fit(data)

    model.write.overwrite().save("./tmp_embeddings_pipeline")

    model.transform(data).show(1)

    val loadedPipeline1 = PipelineModel.load("./tmp_embeddings_pipeline")

    loadedPipeline1.transform(data).show(1)

    val loadedPipeline2 = PipelineModel.load("./tmp_embeddings_pipeline")

    loadedPipeline2.transform(data).show(1)
  }

  "WordEmbeddingsModel" should "not reset sentence indexes of the documents it processes" in {
    val rows = Array(
      Tuple1("""In short, the patient is a 55-year-old entleman with a broken leg. The guy also has long-standing morbid obesity, resistant to nonsurgical methods
      of weight loss with of 69.7 with comorbidities of hypertension, atrial fibrillation,
      hyperlipidemia, possible sleep apnea, and also osteoarthritis of the lower extremities.
      He is also an ex-smoker. Does not have diabetes at all. He is currently smoking SMOKER and he is planning to quit and
      at least he should do this six to eight days before for multiple reasons including decreasing the DVT,
      pulmonary embolism rates and marginal ulcer problems after surgery, which will be discussed later on.""")
    )
    val df = ResourceHelper.spark.createDataFrame(rows).toDF("text")
    val dac = new DocumentAssembler().setInputCol("text").setOutputCol("doc")
    val sd = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")
      .setInputCols("doc").setOutputCol("sentence")
    val tk = new Tokenizer().setInputCols("sentence").setOutputCol("token")
    val emb = WordEmbeddingsModel.pretrained("embeddings_clinical","en","clinical/models").setOutputCol("embs")
    val ner = NerDLModel.pretrained("ner_clinical","en","clinical/models").setInputCols("sentence","token","embs").setOutputCol("ner")
    val conv = new NerConverter().setInputCols("sentence","token","ner").setOutputCol("ner_chunk")
    val c2d = new Chunk2Doc().setInputCols("ner_chunk").setOutputCol("chunk_doc")
    val ctk = new ChunkTokenizer().setInputCols("ner_chunk").setOutputCol("chunk_tk") //Optional


    val cembs = WordEmbeddingsModel.pretrained("embeddings_healthcare_100d","en","clinical/models")
      .setInputCols("chunk_doc","chunk_tk").setOutputCol("chunk_tk_embs")
    val aggembs = new ChunkEmbeddings().setInputCols("ner_chunk","chunk_tk_embs").setOutputCol("chunk_embs")
      .setPoolingStrategy("AVERAGE").setSkipOOV(false)
    val embs_pl = new Pipeline().setStages(Array(dac, sd, tk, emb, ner, conv, c2d, ctk, cembs, aggembs)).fit(df)
    val out_df = embs_pl.transform(df)
    out_df.selectExpr("explode(arrays_zip(ner_chunk.result, chunk_embs.embeddings)) as a").show(100, truncate=50)

    val textSent = out_df.selectExpr("explode(arrays_zip(chunk_tk.result,chunk_tk.metadata,chunk_tk_embs.result,chunk_tk_embs.metadata)) as a")
      .selectExpr("(a['0'],a['1'].sentence) as chunk",
        "(a['2'],a['3'].sentence) as embs").collect().forall(r=>r(0)==r(1))
    assert(textSent)
  }
}
