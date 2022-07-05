# data-communities

The code in this repository is used to analyze a citation network of ICPSR datasets cited in academic literature. The data analyzed is derived from the [ICPSR Bibliography of Data-Related Literature](https://www.icpsr.umich.edu/web/pages/ICPSR/citations/) citations. This analysis supports the paper [Subdivisions and Crossroads: Identifying Hidden Community Structures in a Data Archive’s Citation Network](https://doi.org/10.48550/arXiv.2205.08395). Data are archived on [openICPSR](https://www.openicpsr.org/openicpsr/project/174361/version/V1/view).

[![DOI](https://zenodo.org/badge/492936222.svg)](https://zenodo.org/badge/latestdoi/492936222)

## code/common_functions.py
Functions used for network analysis

## code/community_detection_datasets.ipynb
Analysis notebook for constructing and analyzing communities in the dataset co-citation network

## code/community_detection_fields.ipynb
Analysis notebook for constructing and analyzing communities in the field of research co-citation network

## data/ICPSR_BIBLIOGRAPHY.xlsx
Dataset of ICPSR Bibliography citations collected up to February 2022 (one to many relationship between publication and ICPSR datasets)
