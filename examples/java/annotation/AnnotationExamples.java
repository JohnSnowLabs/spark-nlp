package com.johnsnowlabs.nlp;

import com.johnsnowlabs.nlp.annotators.LemmatizerModel;
import com.johnsnowlabs.nlp.annotators.Tokenizer;
import com.johnsnowlabs.nlp.embeddings.EmbeddingsHelper;
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.LinkedList;

public class AnnotationExamples {

    public static void main(String args[]) {

        DocumentAssembler document = new DocumentAssembler();
        document.setInputCol("text");
        document.setOutputCol("document");
        document.setCleanupMode("disabled");

        Tokenizer tokenizer = new Tokenizer();
        tokenizer.setInputCols(new String[] {"document"});
        tokenizer.setOutputCol("token");

        Pipeline pipeline = new Pipeline();
        pipeline.setStages(new PipelineStage[] {document, tokenizer});

        SparkSession spark = com.johnsnowlabs.nlp.SparkNLP.start();

        LinkedList<String> text = new java.util.LinkedList<String>();

        text.add("Peter is a very good person");

        Dataset<Row> data = spark.createDataset(text, Encoders.STRING()).toDF("text");

        PipelineModel pipelineModel = pipeline.fit(data);

        Dataset<Row> transformed = pipelineModel.transform(data);
        transformed.show();

        PretrainedPipeline pretrained = new PretrainedPipeline("explain_document_dl");
        pretrained.transform(data).show();

        LemmatizerModel lemmatizer = (LemmatizerModel) LemmatizerModel.pretrained("lemma_antbnc");
        lemmatizer.setInputCols(new String[] {"token"});
        lemmatizer.setOutputCol("lemma");

        lemmatizer.transform(transformed).show();

        LightPipeline lightPipeline = new LightPipeline(pipelineModel, true);

        java.util.Map<String, java.util.List<String>> result = lightPipeline.annotateJava("Peter is a very good person.");

        System.out.println(result.get("token"));

        java.util.ArrayList<String> list = new java.util.ArrayList<String>();
        list.add("Peter is a good person.");
        list.add("Roy lives in Germany.");

        System.out.println(lightPipeline.annotateJava(list));

        EmbeddingsHelper.load(
                "./random_embeddings_dim4.txt",
                spark,
                "TEXT",
                "random",
                4,
                false);

        System.out.println("\nFinished testing Spark NLP on JAVA");

    }
}
