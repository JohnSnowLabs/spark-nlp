/*
 * Copyright 2017-2021 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.parser.typdep;

import gnu.trove.map.hash.TObjectIntHashMap;

import java.util.Arrays;
import java.util.List;

public class PredictionParameters {

    private TObjectIntHashMap map;

    public PredictionParameters(){
        map = new TObjectIntHashMap(10000);
    }

    public TObjectIntHashMap transformToTroveMap(String mapAsString){

        List<String> mapAsArray = transformToListOfString(mapAsString);

        for (String keyAndValue : mapAsArray) {
            int index = keyAndValue.lastIndexOf('=');
            if (index > -1){
                String key = keyAndValue.substring(0, index);
                String value = keyAndValue.substring(index+1);
                if (!value.equals("") && value.matches("\\d+")) {
                    this.map.put(key, Integer.parseInt(value));
                }
            }
        }

        return map;
    }

    private List<String> transformToListOfString(String mapAsString){
        String cleanMapAsString =  mapAsString.replace("{","").replace("}","");
        String[] mapAsArray = cleanMapAsString.split(",");
        return Arrays.asList(mapAsArray);
    }

}
