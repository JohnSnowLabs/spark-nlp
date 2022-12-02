/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp;

import com.johnsnowlabs.nlp.annotators.LemmatizerModel;
import com.johnsnowlabs.nlp.annotators.Tokenizer;
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline;
import com.johnsnowlabs.storage.StorageHelper;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.LinkedList;

public class GeneralAnnotationsTest {

    public static void main(String[] args) {

        DocumentAssembler document = new DocumentAssembler();
        document.setInputCol("text");
        document.setOutputCol("document");
        document.setCleanupMode("disabled");

        Tokenizer tokenizer = new Tokenizer();
        tokenizer.setInputCols(new String[]{"document"});
        tokenizer.setOutputCol("token");

        Pipeline pipeline = new Pipeline();
        pipeline.setStages(new PipelineStage[]{document, tokenizer});

        SparkSession spark = com.johnsnowlabs.nlp.SparkNLP.start(
                false,
                false,
                false,
                "16G",
                "",
                "",
                "");

        LinkedList<String> text = new java.util.LinkedList<>();

        text.add("Peter is a very good person");

        Dataset<Row> data = spark.createDataset(text, Encoders.STRING()).toDF("text");

        PipelineModel pipelineModel = pipeline.fit(data);

        Dataset<Row> transformed = pipelineModel.transform(data);
        transformed.show();

        PretrainedPipeline pretrained = new PretrainedPipeline("explain_document_dl");
        pretrained.transform(data).show();

        LemmatizerModel lemmatizer = LemmatizerModel.pretrained("lemma_antbnc");
        lemmatizer.setInputCols(new String[]{"token"});
        lemmatizer.setOutputCol("lemma");

        lemmatizer.transform(transformed).show();

        LightPipeline lightPipeline = new LightPipeline(pipelineModel, true);

        java.util.Map<String, java.util.List<String>> result = lightPipeline.annotateJava("Peter is a very good person.");

        System.out.println(result.get("token"));

        java.util.ArrayList<String> list = new java.util.ArrayList<>();
        list.add("Peter is a good person.");
        list.add("Roy lives in Germany.");

        System.out.println(lightPipeline.annotateJava(list));

        StorageHelper.load(
                "src/test/resources/random_embeddings_dim4.txt",
                spark,
                "random",
                "random_ref",
                false
        );

        System.out.println("\nFinished testing Spark NLP on JAVA");

    }
}
