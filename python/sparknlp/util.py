#  Licensed to the Apache Software Foundation (ASF) under one or more
#  contributor license agreements.  See the NOTICE file distributed with
#  this work for additional information regarding copyright ownership.
#  The ASF licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import sparknlp.internal as _internal


def get_config_path():
    return _internal._ConfigLoaderGetter().apply()


class CoNLLGenerator:
    @staticmethod
    def exportConllFiles(spark, files_path, pipeline, output_path):
        _internal._CoNLLGeneratorExport(spark, files_path, pipeline, output_path).apply()

    @staticmethod
    def exportConllFiles(dataframe, output_path):
        _internal._CoNLLGeneratorExport(dataframe, output_path).apply()
