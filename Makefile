# Binary Paths for MiKTeX (macOS)
PDFLATEX = "/Applications/MiKTeX Console.app/Contents/bin/miktex-pdftex" -undump=pdflatex
BIBTEX = "/Applications/MiKTeX Console.app/Contents/bin/miktex-bibtex"

.PHONY: all setup clean run api test report

all: setup run test

setup:
	@echo "🔧 Setting up environment..."
	pip install -r requirements.txt

run:
	@echo "🚀 Running full pipeline..."
	python3 main.py

api:
	@echo "🌐 Starting FastAPI server..."
	uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

test:
	@echo "🧪 Running full test suite (Invariants + Temporal Integrity)..."
	python3 -m pytest tests/ -v

report:
	@echo "📚 Compiling all deliverables locally..."
	# Compile Executive Summary
	$(PDFLATEX) -interaction=nonstopmode -output-directory=deliverables deliverables/resumen_ejecutivo.tex
	# Compile Retention Contract
	$(PDFLATEX) -interaction=nonstopmode -output-directory=deliverables deliverables/contrato_retencion.tex
	# Compile Technical Report (Multi-pass for BibTeX)
	cp report/references.bib deliverables/
	$(PDFLATEX) -interaction=nonstopmode -output-directory=deliverables deliverables/informe_tecnico.tex
	# Note: bibtex usually needs to run on the .aux file in the output directory
	$(BIBTEX) deliverables/informe_tecnico.aux
	$(PDFLATEX) -interaction=nonstopmode -output-directory=deliverables deliverables/informe_tecnico.tex
	$(PDFLATEX) -interaction=nonstopmode -output-directory=deliverables deliverables/informe_tecnico.tex

clean:
	@echo "🧹 Cleaning up artifacts..."
	rm -rf artifacts/data/*
	rm -rf artifacts/models/*
	rm -rf artifacts/figures/*
	rm -f artifacts/manifest.json
	find . -type d -name "__pycache__" -exec rm -rf {} +
