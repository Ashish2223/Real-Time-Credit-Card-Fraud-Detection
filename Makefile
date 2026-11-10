.PHONY: install train test api simulate sql lint docker

install:
	pip install -e ".[dev]"

train:
	python -m fraud_detection.train

test:
	pytest -q

api:
	uvicorn fraud_detection.api.main:app --reload --port 8000

simulate:
	python scripts/simulate_stream.py --n 50

sql:
	python scripts/sql_analysis.py --load

lint:
	ruff check .

docker:
	docker build -t fraud-api . && docker run -p 8000:8000 fraud-api
