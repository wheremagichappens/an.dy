.PHONY: all etl label train score eval monitor clean

all: etl label train score eval causal monitor

etl:
	python etl/build_customer_table.py

label:
	python labels/build_churn_label.py

train:
	python models/train_model.py

score:
	python models/score_customers.py

eval:
	python evaluation/intervention_analysis.py
	python evaluation/causal_checks.py

monitor:
	python monitoring/data_drift.py
	python monitoring/score_stability.py

clean:
	rm -rf artifacts
	rm -rf data/processed


causal:
	python causal/build_country_panel.py
	python causal/diff_in_diff.py
	python causal/parallel_trends_plot.py
	python causal/cuped_analysis.py
