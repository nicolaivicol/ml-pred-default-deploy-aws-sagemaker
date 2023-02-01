#!/bin/bash
conda activate klrn-hw
ipynb-py-convert solution_report.py artifacts/solution_report.ipynb
papermill artifacts/solution_report.ipynb artifacts/solution_report.ipynb
jupyter nbconvert artifacts/solution_report.ipynb --to=html --TemplateExporter.exclude_input=True
