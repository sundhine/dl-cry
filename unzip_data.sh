#!/bin/bash

gzip -cd data/te.gz > data/te.csv
gzip -cd data/tr.gz > data/tr.csv
gzip -cd data/va.gz > data/va.csv