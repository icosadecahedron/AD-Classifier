.PHONY: install train test serve docker-build docker-run lint

install:
	pip install -r requirements.txt

train:
	python -m src.pipeline

test:
	pytest tests/ -v

serve:
	uvicorn src.api.main:app --reload --port 8000

docker-build:
	docker build -t ad-classifier .

docker-run:
	docker run -p 8000:8000 ad-classifier

lint:
	ruff check src/ tests/
