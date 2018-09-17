package com.company;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.TreeSet;
import java.util.stream.Collectors;

public class SimilarImageService {

    private INDArray vectors;
    private List<String> images;

    public SimilarImageService(INDArray vectors, List<String> images) {
        this.vectors = vectors;
        this.images = images;
    }

    public List<String> find(int imageIndex, int limit){
        INDArray result = vectors.getColumn(imageIndex).transpose().mmul(vectors);

        return findTopIndexes(result, limit)
                .stream()
                .map( index -> images.get(index))
                .collect(Collectors.toList());
    }

    private List<Integer> findTopIndexes(INDArray prods, int limit){
        TreeSet<IndexWeight> top = new TreeSet<>(new IndexWeightComparator());

        for (int i = 0; i < prods.length(); i++) {
            if (top.size() < limit) {
                top.add(new IndexWeight(i, prods.getDouble(i)));
                continue;
            }

            if (prods.getDouble(i) < top.last().getWeight())
                continue;

            top.add(new IndexWeight(i, prods.getDouble(i)));

            top.pollLast();
        }

        return top.stream().map(index -> index.getIndex()).collect(Collectors.toList());
    }
}
