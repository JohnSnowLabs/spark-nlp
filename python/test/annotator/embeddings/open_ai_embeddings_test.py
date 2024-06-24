import unittest
from sparknlp.annotator import *
from sparknlp.base import *
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession

class OpenAIEmbeddingsTestCase(unittest.TestCase):
# Set your OpenAI API key to run unit test...
    def setUp(self):
        self.spark = SparkSession.builder \
            .appName("Tests") \
            .master("local[*]") \
            .config("spark.driver.memory","8G") \
            .config("spark.driver.maxResultSize", "2G") \
            .config("spark.jars", "lib/sparknlp.jar") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.kryoserializer.buffer.max", "1000m") \
            .config("spark.jsl.settings.openai.api.key","") \
            .getOrCreate()

    def test_openai_embeddings(self):

        documentAssembler = DocumentAssembler() \
                 .setInputCol("text") \
                 .setOutputCol("document")
        openai_embeddings = OpenAIEmbeddings() \
                 .setInputCols("document") \
                 .setOutputCol("embeddings") \
                 .setModel("text-embedding-ada-002")

        import tempfile
        openai_embeddings.write().overwrite().save("file:///" + tempfile.gettempdir() + "/openai_embeddings")
        loaded = OpenAIEmbeddings.load("file:///" + tempfile.gettempdir() + "/openai_embeddings")

        pipeline = Pipeline().setStages([
             documentAssembler,
            loaded
         ])

        sample_text = [["The food was delicious and the waiter..."]]
        sample_df = self.spark.createDataFrame(sample_text).toDF("text")
        pipeline.fit(sample_df).transform(sample_df).select("embeddings").show(truncate=False)



if __name__ == '__main__':
    unittest.main()
