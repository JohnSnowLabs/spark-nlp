#  Copyright 2017-2024 John Snow Labs
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
from sparknlp.internal import ExtendedJavaWrapper


class SparkNLPReader(ExtendedJavaWrapper):
    """Instantiates class to read HTML files.

    Two types of input paths are supported,

    htmlPath: this is a path to a directory of HTML files or a path to an HTML file
    E.g. "path/html/files"

    url: this is the URL or set of URLs of a website  . E.g., "https://www.wikipedia.org"

    Parameters
    ----------
    params : spark
        Spark session
    params : dict, optional
        Parameter with custom configuration

    Examples
    --------
    >>> from sparknlp.reader import SparkNLPReader
    >>> html_df = SparkNLPReader().html(spark, "https://www.wikipedia.org")
    >>> html_df.show(truncate=False)
    +--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |url                 |html                                                                                                                                                                                                                                                                                                                            |
    +--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |https://example.com/|[{Title, 0, Example Domain, {pageNumber -> 1}}, {NarrativeText, 0, This domain is for use in illustrative examples in documents. You may use this domain in literature without prior coordination or asking for permission., {pageNumber -> 1}}, {NarrativeText, 0, More information... More information..., {pageNumber -> 1}}]|
    +--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

    >>> import sparknlp
    >>> html_df = sparknlp.read().html("https://www.wikipedia.org")
    >>> html_df.show(truncate=False)

    """

    def __init__(self, spark, params=None):
        if params is None:
            params = {}
        super(SparkNLPReader, self).__init__("com.johnsnowlabs.reader.SparkNLPReader", params)
        self.spark = spark

    def html(self, htmlPath):
        jdf = self._java_obj.html(htmlPath)
        dataframe = self.getDataFrame(self.spark, jdf)
        return dataframe