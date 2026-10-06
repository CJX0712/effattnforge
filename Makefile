.PHONY: install lint test bench ci all clean

install:
	pip install -r requirements.lock

lint:
	python -m ruff check .

test:
	python -m pytest -q

bench:
	python examples/run_demo.py

ci: lint test

all: lint test bench

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ */__pycache__
