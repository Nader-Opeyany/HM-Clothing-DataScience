#!/bin/bash
fileName=$1
pdflatex "$fileName.tex"
bibtex "$fileName.aux"
pdflatex "$fileName.tex"
pdflatex "$fileName.tex"
