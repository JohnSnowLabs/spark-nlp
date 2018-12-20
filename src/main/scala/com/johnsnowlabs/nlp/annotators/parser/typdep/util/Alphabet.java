package com.johnsnowlabs.nlp.annotators.parser.typdep.util;

import gnu.trove.map.hash.TLongIntHashMap;

import java.io.Serializable;

public class Alphabet implements Serializable
{
    // Serialization
    private static final long serialVersionUID = 1;
    private TLongIntHashMap map;
    private int numEntries;
    private boolean growthStopped = false;

    private Alphabet (int capacity)
    {
        this.map = new TLongIntHashMap(capacity);
        numEntries = 0;
    }

    public Alphabet ()
    {
        this (10000);
    }

    /** Return -1 if entry isn't present. */
    public int lookupIndex (long entry, int value)
    {
        int ret = map.get(entry);
        if (ret <= 0 && !growthStopped) {
            numEntries++;
            ret = value + 1;
            map.put (entry, ret);
        }
        return ret - 1;	// feature id should be 0-based
    }

    /** Return -1 if entry isn't present. */
    public int lookupIndex (long entry)
    {
        int ret = map.get(entry);
        if (ret <= 0 && !growthStopped) {
            numEntries++;
            ret = numEntries;
            map.put (entry, ret);
        }
        return ret - 1;	// feature id should be 0-based
    }

    public void stopGrowth ()
    {
        growthStopped = true;
    }
}
