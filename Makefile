PY := python
PYTHONPATH := $(CURDIR)/src

.PHONY: i a index ask web

# Short aliases with sensible defaults
i: index
a: ask

index:
	@PYTHONPATH="$(PYTHONPATH)" COLLECTION="$(or $(COLLECTION),combined_docs)" QA="$(QA)" QA_MODE="$(or $(QA_MODE),concat)" CLEAR="$(or $(CLEAR),0)" DATA_DIR="$(or $(DATA_DIR),./data)" $(PY) scripts/build_index.py

ask:
	@PYTHONPATH="$(PYTHONPATH)" COLLECTION="$(or $(COLLECTION),combined_docs)" QUESTION="$(QUESTION)" K="$(or $(K),5)" $(PY) scripts/ask.py

web:
	@PYTHONPATH="$(PYTHONPATH)" COLLECTION="$(or $(COLLECTION),combined_docs)" PORT="$(or $(PORT),3000)" FLASK_DEBUG="$(or $(FLASK_DEBUG),false)" $(PY) api/app.py


