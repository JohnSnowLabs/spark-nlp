from sparknlp.functions import *
from sparknlp.base import * 
from sparknlp.annotator import *  
from sparknlp import start
from sparknlp.annotation import Annotation

spark = start()           

documentAssembler = DocumentAssembler() \
.setInputCol('text') \
.setOutputCol('document')                                                                                     

df = spark.createDataFrame([["Pepito clavo un clavillo"], ["Un clavillo muy pillo"]]).toDF("text")               
df = documentAssembler.transform(df)                                                                             
mapped = map_annotations_col(df.select("document"), lambda x: [a.copy(a.result.lower()) for a in x], "document", "text_tail", Annotation.arrayType())
mapped.show(truncate=False)
