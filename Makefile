.PHONY: test-flask test-integration

TAG="\n\n\033[0;32m\#\#\# "
END=" \#\#\# \033[0m\n"

test-flask:
	@echo $(TAG)Running tests$(END)
	PYTHONPATH=. py.test -s tests/flask_tests.py

test-integration:
	@echo $(TAG)Running tests$(END)
	PYTHONPATH=. py.test -s tests/integration_tests.py
