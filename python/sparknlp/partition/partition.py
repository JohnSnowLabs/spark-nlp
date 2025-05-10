#  Copyright 2017-2025 John Snow Labs
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import sparknlp
from sparknlp.internal import ExtendedJavaWrapper

"""
     The Partition class provides a streamlined and user-friendly interface for interacting with
     Spark NLP readers. It allows you to extract content from various file formats while providing
     customization using keyword arguments. File types include Email, Excel, HTML, PPT,
     Text, Word documents.
    
    Parameters
    ----------
    params : dict, optional   
        Parameter with custom configuration
        
    Examples
    ----------
    txt_directory = "/content/txtfiles/reader/txt"
    partition_df = Partition(content_type = "text/plain").partition(txt_directory)
    partition_df.show()
    
     +--------------------+--------------------+
     |                path|                 txt|
     +--------------------+--------------------+
     |file:/content/txt...|[{Title, BIG DATA...|
     +--------------------+--------------------+

"""

class Partition(ExtendedJavaWrapper):

    def  __init__(self, **kwargs):
        self.spark = sparknlp.start()
        params = {}
        for key, value in kwargs.items():
            try:
                params[key] = str(value)
            except Exception as e:
                raise ValueError(f"Invalid value for key '{key}': Cannot cast {type(value)} to string. Original error: {e}")

        super(Partition, self).__init__("com.johnsnowlabs.partition.Partition", params)

    """
        Takes a URL/file/directory path to read and parse it's content.
        
        Parameters
        ----------
        path : string   
            Path to a file or local directory where all files are stored. Supports URLs and DFS file systems like databricks, HDFS and Microsoft Fabric OneLake.
        headers: dict, optional
            If the path is a URL it sets the necessary headers for the request.
        
        Returns
        -------
        DataFrame
            DataFrame with parsed file content.
    """

    def partition(self, path, headers=None):
        if headers is None:
            headers = {}
        jdf = self._java_obj.partition(path, headers)
        dataframe = self.getDataFrame(self.spark, jdf)
        return dataframe

    """
        Parses and reads data from multiple URL's.
        
        Parameters
        ----------
        urls : List[str] 
            list of URL's
        headers: dict, optional
            sets the necessary headers for the URL request.
        
        Returns
        -------
        DataFrame
            DataFrame with parsed url content.
    """
    def partition_urls(self, path, headers=None):
        if headers is None:
            headers = {}
        jdf = self._java_obj.partitionUrlsJava(path, headers)
        dataframe = self.getDataFrame(self.spark, jdf)
        return dataframe

    """
        Partitions text from a string.
        
        Parameters
        ----------
        text : string
            text data in string form 
            
        Returns
        -------
        DataFrame
            DataFrame with parsed text content.
    """
    def partition_text(self, text):
        jdf = self._java_obj.partitionText(text)
        dataframe = self.getDataFrame(self.spark, jdf)
        return dataframe