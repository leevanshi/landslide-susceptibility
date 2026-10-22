# Makefile for Landslide Susceptibility AI Project

.PHONY: help train ask app report test data clean

help:
	@echo "Available targets:"
	@echo "  make train   - Run training, spatial cross-validation, calibration, SHAP, and map export"
	@echo "  make ask     - Launch interactive query analyzer"
	@echo "  make app     - Start Streamlit interactive dashboard"
	@echo "  make report  - Generate HTML and PDF assessment reports"
	@echo "  make test    - Run pytest suite (parser, spatial split, scenario simulator, loader)"
	@echo "  make data    - Generate synthetic benchmark dataset"

train:
	python train.py

ask:
	python analyze.py

app:
	streamlit run app.py

report:
	python report.py

test:
	python -m pytest

data:
	python generate_data.py
