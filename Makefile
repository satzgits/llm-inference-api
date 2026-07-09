.PHONY: install run test docker clean

install:
	pip install -r requirements.txt

run:
	uvicorn app.main:app --reload --port 8000

test:
	python -m pytest tests/ -v

clean:
	rm -rf __pycache__ app/__pycache__ tests/__pycache__

docker:
	docker build -t llm-inference-api .
