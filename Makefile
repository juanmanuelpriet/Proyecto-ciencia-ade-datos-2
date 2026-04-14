# =============================================================================
# Makefile — Global-E-Shop Churn Predictor
# Compatible with macOS (Homebrew/MacTeX) and Linux (apt pdflatex)
# =============================================================================

# ── LaTeX toolchain for MiKTeX (macOS) ──────────────────────────────────────
PDFLATEX = "/Applications/MiKTeX Console.app/Contents/bin/miktex-pdftex" -undump=pdflatex
BIBTEX   = "/Applications/MiKTeX Console.app/Contents/bin/miktex-bibtex"
DELIVERABLES_DIR = deliverables

.PHONY: all setup clean run api test report baselines seed help

# ─────────────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Global-E-Shop Churn Predictor — Comandos disponibles"
	@echo "  ════════════════════════════════════════════════════"
	@echo "  make setup      → Instalar dependencias (pip install -r requirements.txt)"
	@echo "  make run        → Ejecutar pipeline completo (seed + ETL + EDA + ML)"
	@echo "  make seed       → Solo regenerar datos sintéticos"
	@echo "  make api        → Iniciar servidor FastAPI en localhost:8000"
	@echo "  make test       → Ejecutar suite de pruebas (pytest tests/ -v)"
	@echo "  make baselines  → Evaluar baselines honestos standalone"
	@echo "  make report     → Compilar entregables LaTeX → PDF"
	@echo "  make clean      → Borrar artefactos generados (datos, modelos, figuras)"
	@echo "  make all        → setup + run + test (pipeline completo)"
	@echo ""

# ─── Pipeline ─────────────────────────────────────────────────────────────────
all: setup run test

setup:
	@echo "🔧 Instalando dependencias..."
	pip install -r requirements.txt

seed:
	@echo "🌱 Regenerando datos sintéticos..."
	python3 main.py --force-seed --skip-train

run:
	@echo "🚀 Ejecutando pipeline completo..."
	python3 main.py

api:
	@echo "🌐 Iniciando FastAPI en http://localhost:8000 ..."
	@echo "   Documentación: http://localhost:8000/docs"
	uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# ─── Tests ────────────────────────────────────────────────────────────────────
test:
	@echo "🧪 Ejecutando suite completa de pruebas..."
	python3 -m pytest tests/ -v --tb=short

baselines:
	@echo "📏 Evaluando baselines honestos..."
	python3 ml/baselines.py

# ─── LaTeX → PDF ──────────────────────────────────────────────────────────────
# Requiere pdflatex en PATH:
#   macOS (MacTeX):  brew install --cask mactex  OR  brew install basictex
#   Linux (TeX Live): sudo apt-get install texlive-full
report:
	@echo "📚 Compilando entregables LaTeX → PDF..."
	@command -v $(PDFLATEX) >/dev/null 2>&1 || \
		{ echo "❌ pdflatex no encontrado. Verifique la ruta en el Makefile."; exit 1; }

	@echo "   → Sincronizando fuentes desde report/ ..."
	cp report/informe_tecnico.tex deliverables/
	cp report/references.bib deliverables/

	@echo "   → Compilando contrato_retencion.tex ..."
	cd $(DELIVERABLES_DIR) && \
		$(PDFLATEX) -interaction=nonstopmode contrato_retencion.tex > /dev/null && \
		$(PDFLATEX) -interaction=nonstopmode contrato_retencion.tex > /dev/null
	@echo "   ✅ contrato_retencion.pdf generado"

	@echo "   → Compilando resumen_ejecutivo.tex ..."
	cd $(DELIVERABLES_DIR) && \
		$(PDFLATEX) -interaction=nonstopmode resumen_ejecutivo.tex > /dev/null && \
		$(PDFLATEX) -interaction=nonstopmode resumen_ejecutivo.tex > /dev/null
	@echo "   ✅ resumen_ejecutivo.pdf generado"

	@echo "   → Compilando informe_tecnico.tex (4 pasadas para índice/bib) ..."
	cd $(DELIVERABLES_DIR) && \
		$(PDFLATEX) -interaction=nonstopmode informe_tecnico.tex > /dev/null && \
		$(BIBTEX) informe_tecnico > /dev/null || true && \
		$(PDFLATEX) -interaction=nonstopmode informe_tecnico.tex > /dev/null && \
		$(PDFLATEX) -interaction=nonstopmode informe_tecnico.tex > /dev/null
	@echo "   ✅ informe_tecnico.pdf generado (con índice y referencias)"

	@echo ""
	@echo "📄 PDFs finales disponibles en: $(DELIVERABLES_DIR)/"
	@ls -lh $(DELIVERABLES_DIR)/*.pdf 2>/dev/null || echo "(no se encontraron PDFs)"

# ─── Limpieza ─────────────────────────────────────────────────────────────────
clean:
	@echo "🧹 Limpiando artefactos generados..."
	rm -rf artifacts/data/*
	rm -rf artifacts/models/*
	rm -rf artifacts/figures/*
	rm -f  artifacts/manifest.json
	# LaTeX auxiliares en deliverables/
	rm -f  $(DELIVERABLES_DIR)/*.aux \
	       $(DELIVERABLES_DIR)/*.log \
	       $(DELIVERABLES_DIR)/*.out \
	       $(DELIVERABLES_DIR)/*.toc \
	       $(DELIVERABLES_DIR)/*.bbl \
	       $(DELIVERABLES_DIR)/*.blg \
	       $(DELIVERABLES_DIR)/*.fdb_latexmk \
	       $(DELIVERABLES_DIR)/*.fls \
	       $(DELIVERABLES_DIR)/*.synctex.gz \
	       $(DELIVERABLES_DIR)/*.lof \
	       $(DELIVERABLES_DIR)/*.lot
	# Cachés de Python
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null; true
	@echo "✅ Limpieza completada."
