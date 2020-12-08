package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.annotator.{ChunkTokenizer, Chunker, NerConverter, NerDLModel, PerceptronModel, SentenceDetectorDLModel}
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.{NGramGenerator, StopWordsCleaner, Tokenizer}
import com.johnsnowlabs.nlp.base.{DocumentAssembler, RecursivePipeline}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{AnnotatorBuilder, Chunk2Doc, EmbeddingsFinisher, Finisher}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.size
import org.scalatest._

class ChunkEmbeddingsTestSpec extends FlatSpec {

  "ChunkEmbeddings" should "correctly calculate chunk embeddings from Chunker" in {

    val smallCorpus = ResourceHelper.spark.read.option("header","true").csv("src/test/resources/embeddings/sentence_embeddings.csv")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val posTagger = PerceptronModel.pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("pos")

    val chunker= new Chunker()
      .setInputCols(Array("sentence", "pos"))
      .setOutputCol("chunk")
      .setRegexParsers(Array("<DT>?<JJ>*<NN>+"))

    val embeddings = AnnotatorBuilder.getGLoveEmbeddings(smallCorpus)
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)

    val chunkEmbeddings = new ChunkEmbeddings()
      .setInputCols(Array("chunk", "embeddings"))
      .setOutputCol("chunk_embeddings")
      .setPoolingStrategy("AVERAGE")

    val sentenceEmbeddingsChunk = new SentenceEmbeddings()
      .setInputCols(Array("document", "chunk_embeddings"))
      .setOutputCol("sentence_embeddings_chunks")
      .setPoolingStrategy("AVERAGE")

    val finisher = new EmbeddingsFinisher()
      .setInputCols("chunk_embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsVector(true)
      .setCleanAnnotations(false)

    val pipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        tokenizer,
        posTagger,
        chunker,
        embeddings,
        chunkEmbeddings,
        sentenceEmbeddingsChunk,
        finisher
      ))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)

//    pipelineDF.select("chunk.metadata").show(2)
//    pipelineDF.select("chunk.result").show(2)
//
//    pipelineDF.select("embeddings.metadata").show(2)
//    pipelineDF.select("embeddings.embeddings").show(2)
//    pipelineDF.select("embeddings.result").show(2)
//
//    pipelineDF.select("chunk_embeddings").show(2)
//    println("Chunk Embeddings")
//    pipelineDF.select("chunk_embeddings.embeddings").show(2)
//    pipelineDF.select(size(pipelineDF("chunk_embeddings.embeddings")).as("chunk_embeddings_size")).show
//
//    pipelineDF.select("sentence_embeddings_chunks.embeddings").show(2)
//    pipelineDF.select(size(pipelineDF("sentence_embeddings_chunks.embeddings")).as("chunk_embeddings_size")).show
//
//    pipelineDF.select("finished_embeddings").show(2)
//    pipelineDF.select(size(pipelineDF("finished_embeddings")).as("chunk_embeddings_size")).show

  }

  "ChunkEmbeddings" should "correctly calculate chunk embeddings from NGramGenerator" in {

    val smallCorpus = ResourceHelper.spark.read.option("header","true").csv("src/test/resources/embeddings/sentence_embeddings.csv")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val nGrams = new NGramGenerator()
      .setInputCols("token")
      .setOutputCol("chunk")
      .setN(2)

    val embeddings = AnnotatorBuilder.getGLoveEmbeddings(smallCorpus)
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)

    val chunkEmbeddings = new ChunkEmbeddings()
      .setInputCols(Array("chunk", "embeddings"))
      .setOutputCol("chunk_embeddings")
      .setPoolingStrategy("AVERAGE")

    val finisher = new Finisher()
      .setInputCols("chunk_embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsArray(true)
      .setCleanAnnotations(false)

    val pipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        tokenizer,
        nGrams,
        embeddings,
        chunkEmbeddings,
        finisher
      ))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)

//    pipelineDF.select("token.metadata").show(2)
//    pipelineDF.select("token.result").show(2)
//
//    pipelineDF.select("chunk.metadata").show(2)
//    pipelineDF.select("chunk.result").show(2)
//
//    pipelineDF.select("embeddings.metadata").show(2)
//    pipelineDF.select("embeddings.embeddings").show(2)
//    pipelineDF.select("embeddings.result").show(2)
//
//    pipelineDF.select("chunk_embeddings").show(2)
//
//    pipelineDF.select("chunk_embeddings.embeddings").show(2)
//    pipelineDF.select(size(pipelineDF("chunk_embeddings.embeddings")).as("chunk_embeddings_size")).show
//
//    pipelineDF.select("finished_embeddings").show(2)
//    pipelineDF.select(size(pipelineDF("finished_embeddings")).as("chunk_embeddings_size")).show

    assert(pipelineDF.selectExpr("explode(chunk_embeddings.metadata) as meta").select("meta.chunk").distinct().count() > 1)
  }

  "ChunkEmbeddings" should "correctly work with empty tokens" in {

    val smallCorpus = ResourceHelper.spark.read.option("header","true").csv("src/test/resources/embeddings/sentence_embeddings.csv")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val stopWordsCleaner = new StopWordsCleaner()
      .setInputCols("token")
      .setOutputCol("cleanTokens")
      .setStopWords(Array("this", "is", "my", "document", "sentence", "second", "first", ",", "."))
      .setCaseSensitive(false)

    val posTagger = PerceptronModel.pretrained()
      .setInputCols("sentence", "cleanTokens")
      .setOutputCol("pos")

    val chunker= new Chunker()
      .setInputCols(Array("sentence", "pos"))
      .setOutputCol("chunk")
      .setRegexParsers(Array("<DT>?<JJ>*<NN>+"))

    val embeddings = AnnotatorBuilder.getGLoveEmbeddings(smallCorpus)
      .setInputCols("sentence", "cleanTokens")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)

    val chunkEmbeddings = new ChunkEmbeddings()
      .setInputCols(Array("chunk", "embeddings"))
      .setOutputCol("chunk_embeddings")
      .setPoolingStrategy("AVERAGE")
      .setSkipOOV(true)

    val embeddingsSentence = new SentenceEmbeddings()
      .setInputCols(Array("sentence", "chunk_embeddings"))
      .setOutputCol("sentence_embeddings")
      .setPoolingStrategy("AVERAGE")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        tokenizer,
        stopWordsCleaner,
        posTagger,
        chunker,
        embeddings,
        chunkEmbeddings,
        embeddingsSentence
      ))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)
    println(pipelineDF.count())
    pipelineDF.show()
    pipelineDF.select("chunk").show(1)
    pipelineDF.select("embeddings").show(1)
    pipelineDF.select("sentence_embeddings").show(1)

  }

  "ChunkEmbeddings" should "correctly work with multiple chunks per sentence" in {
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
    val sd = SentenceDetectorDLModel.pretrained("sentence_detector_dl","en")
      .setInputCols("doc").setOutputCol("sentence")
    val tk = new Tokenizer().setInputCols("sentence").setOutputCol("token")
    val emb = WordEmbeddingsModel.pretrained("glove_100d").setOutputCol("embs")
    val ner = NerDLModel.pretrained("ner_dl").setInputCols("sentence","token","embs").setOutputCol("ner")
    val conv = new NerConverter().setInputCols("sentence","token","ner").setOutputCol("ner_chunk")
    val c2d = new Chunk2Doc().setInputCols("ner_chunk").setOutputCol("chunk_doc")
    val ctk = new ChunkTokenizer().setInputCols("ner_chunk").setOutputCol("chunk_tk") //Optional
    val cembs = WordEmbeddingsModel.pretrained("glove_6B_300", "xx")
      .setInputCols("chunk_doc","chunk_tk").setOutputCol("chunk_tk_embs")
    val aggembs = new ChunkEmbeddings().setInputCols("ner_chunk","chunk_tk_embs").setOutputCol("chunk_embs")
      .setPoolingStrategy("AVERAGE").setSkipOOV(false)
    val embs_pl = new Pipeline().setStages(Array(dac, sd, tk, emb, ner, conv, c2d, ctk, cembs, aggembs)).fit(df)
    val out_df = embs_pl.transform(df)
    out_df.selectExpr("explode(arrays_zip(ner_chunk.result, chunk_embs.embeddings)) as a").show(100, truncate=50)

    val textSent = out_df.selectExpr("explode(arrays_zip(chunk_tk.result,chunk_tk.metadata,chunk_tk_embs.result,chunk_tk_embs.metadata)) as a")
    .selectExpr("(a['0'],a['1'].sentence) as chunk",
      "(a['2'],a['3'].sentence) as embs")
    textSent.show(100,false)
     val assertion = textSent.collect().forall(r=>r(0)==r(1))
    assert(assertion)
  }

}
