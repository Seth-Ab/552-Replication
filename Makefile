.PHONY: all clean

all: paper/paper.pdf

output/tables/main_result.tex \
output/tables/table3_5fold_full_results.csv \
output/tables/table3_5fold_theta_by_rep.csv \
output/tables/table3_5fold_se_by_rep.csv: simple.py input/sipp1991.dta
	python3 simple.py

paper/paper.pdf: paper/paper.tex paper/references.bib output/tables/main_result.tex
	cd paper && pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex

clean:
	rm -f output/tables/table3_5fold_full_results.csv
	rm -f output/tables/table3_5fold_theta_by_rep.csv
	rm -f output/tables/table3_5fold_se_by_rep.csv
	rm -f output/tables/main_result.tex
	rm -f paper/paper.pdf paper/paper.aux paper/paper.log paper/paper.out paper/paper.bbl paper/paper.blg
