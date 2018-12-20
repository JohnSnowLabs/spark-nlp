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
                if (!value.equals("")){
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
