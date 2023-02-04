/**
 * Windows 10
 * Spark 2.4.3
 * Spark-NLP 2.2.1
 * Hadoop 2.7.2
 *
 * For more details please check-out built.sbt
 * Actually this example was part of a rest application built on Play-Framework, few extra dependencies will be there.
 *
 * Please note that paths needs to be changed in the below source code: currently all path are absolute (NFS & not hdfs)
 */
object CustomForNerDLPipeline{

        // Flag to enable training or use the saved model. Initially keep it as true
        // and when model gets saved to provided location, we can simply load the same
        val ENABLE_TRAINING=true

        // word embedding dimensions: each word would be a vector with given length
        val EMBEDDING_DIMENSIONS=300

        // The spark-nlp library has few defined configurations and if someone needs a different configuration
        // then one has to create a graph with required configuration
        // Please go through this link for more details: https://nlp.johnsnowlabs.com/docs/en/graph
        val PATH_TO_GRAPH_FOLDER="C:\\OpenSourceData\\GRAPH_FOLDER"

        // we have used glove word embeddings, one can learn word-embeddings related to it's data but it works fine.
        val PATH_TO_EXTERAL_EMBEDDINGS_SOURCE="file:///C:/OpenSourceData/REFERENTIAL_DATA/glove.6B.300d.txt"

        // Path to saved pipeline (we didn't just save model, we are saving the entire pipeline)
        val PATH_TO_TRAINED_SAVED_PIPELINE="file:///C:/OpenSourceData/SAVED_MODELS/PreprocessedDummyEmailsData.pipeline"

        // Tagged Data in ConLL-format
        val PATH_TO_EXTERNAL_DATA__TO_BE_USED_FOR_TRAINING="file:///C:/OpenSourceData/input/spark-nlp/TaggedPreprocessedDummyDataOfEmails.conll"


        def main(args:Array[String]):Unit={
        Logger.getLogger("org").setLevel(Level.OFF)
        Logger.getLogger("akka").setLevel(Level.OFF)
        Logger.getRootLogger.setLevel(Level.INFO)

        val spark:SparkSession=SparkSession.builder().appName("test").master("local[*]")
        .config("spark.driver.memory","12G").config("spark.kryoserializer.buffer.max","200M")
        .config("spark.serializer","org.apache.spark.serializer.KryoSerializer").getOrCreate()
        spark.sparkContext.setLogLevel("FATAL")

        val document=new DocumentAssembler().setInputCol("text").setOutputCol("document")

        val token=new Tokenizer().setInputCols("document").setOutputCol("token")

        val word_embeddings=new WordEmbeddings().setInputCols(Array("document","token")).setOutputCol("word_embeddings")
        .setEmbeddingsSource(PATH_TO_EXTERAL_EMBEDDINGS_SOURCE,EMBEDDING_DIMENSIONS,WordEmbeddingsFormat.TEXT)

        val trainingConll=CoNLL().readDataset(spark,PATH_TO_EXTERNAL_DATA__TO_BE_USED_FOR_TRAINING)

        val ner=new NerDLApproach()
        .setInputCols("document","token","word_embeddings")
        .setOutputCol("ner")
        .setLabelColumn("label")
        .setMaxEpochs(120)
        .setRandomSeed(0)
        .setPo(0.03f)
        .setLr(0.2f)
        .setDropout(0.5f)
        .setBatchSize(9)
        .setGraphFolder(PATH_TO_GRAPH_FOLDER)
        .setVerbose(Verbose.Epochs)


        val nerConverter=new NerConverter().setInputCols("document","token","ner").setOutputCol("ner_converter")

        val finisher=new Finisher().setInputCols("ner","ner_converter").setIncludeMetadata(true).setOutputAsArray(false)
        .setCleanAnnotations(false).setAnnotationSplitSymbol("@").setValueSplitSymbol("#")

        val pipeline=new Pipeline()
        .setStages(Array(document,token,word_embeddings,ner,nerConverter,finisher))

        val testingForTop10Carriers=Seq(
        (1,"Google has announced the release of a beta version of the popular TensorFlow machine learning library"),
        (2,"The Paris metro will soon enter the 21st century, ditching single-use paper tickets for rechargeable electronic cards.")
        ).toDS.toDF("_id","text")

        val testing=testingForTop10Carriers
        var pipelineModel:PipelineModel=null
        if(ENABLE_TRAINING){
        println("Training started.......")
        pipelineModel=pipeline.fit(trainingConll)
        pipelineModel.write.save(PATH_TO_TRAINED_SAVED_PIPELINE)
        println(s"Pipeline Model saved '$TRAINED_PIPELINE_NAME'.........")
        }
        else{
        println(s"Loading the already built model from '$TRAINED_PIPELINE_NAME'.........")
        pipelineModel=PipelineModel.load(PATH_TO_TRAINED_SAVED_PIPELINE)
        }

        val result=pipelineModel.transform(testing)

        result.select("ner_converter")show(truncate=false)

        val actualListOfNamedEntitiesMap=result.select("finished_ner").collectAsList().toArray
        .map(x=>x.toString.drop(1).dropRight(1).split("@")).map(keyValuePair=>keyValuePair
        .map(x=>(x.split("->").lastOption.get,x.slice(x.indexOf("->")+2,x.indexOf("#")))).filter(!_._1.equals("O"))
        .groupBy(_._1).mapValues(_.map(_._2).toList))


        val length=actualListOfNamedEntitiesMap.length
        for(index<-0until length){
        println("Keys present in actualOutputMap but not in actualOutputMap:  %s".format(actualListOfNamedEntitiesMap(index)))
        }

        }
        }
