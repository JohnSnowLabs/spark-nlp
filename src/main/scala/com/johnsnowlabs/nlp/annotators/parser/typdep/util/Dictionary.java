package com.johnsnowlabs.nlp.annotators.parser.typdep.util;

import gnu.trove.map.hash.TObjectIntHashMap;

import java.io.Serializable;

public class Dictionary implements Serializable
{

    // Serialization
    private static final long serialVersionUID = 1;

    private TObjectIntHashMap map;

    public TObjectIntHashMap getMap() {
        return map;
    }

    public void setMap(TObjectIntHashMap map) {
        this.map = map;
    }

    private int numEntries;
    private boolean growthStopped = false;
    private String mapAsString;

    public int getNumEntries() {
        return numEntries;
    }

    public boolean isGrowthStopped() {
        return growthStopped;
    }

    public void setNumEntries(int numEntries) {
        this.numEntries = numEntries;
    }

    public void setGrowthStopped(boolean growthStopped) {
        this.growthStopped = growthStopped;
    }

    public void setMapAsString(String mapAsString) {
        this.mapAsString = mapAsString;
    }

    public String getMapAsString() {
        return mapAsString;
    }

    private Dictionary (int capacity)
    {
        this.map = new TObjectIntHashMap(capacity);
        numEntries = 0;
    }

    Dictionary ()
    {
        this (10000);
    }

    /** Return -1 (in old trove version) or 0 (in trove current verion) if entry isn't present. */
    public int lookupIndex (Object entry)
    {
        if (entry == null)
            throw new IllegalArgumentException ("Can't lookup \"null\" in an Alphabet.");
        int ret = map.get(entry);
        if (ret <= 0 && !growthStopped) {
            numEntries++;
            ret = numEntries;
            map.put(entry, ret);
        }
        return ret;
    }

    public Object[] toArray () {
        return map.keys();
    }

    public int dictionarySize()
    {
        return numEntries;
    }

    void stopGrowth ()
    {
        growthStopped = true;
    }

}
