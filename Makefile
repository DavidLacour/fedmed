# One-command entry points. `make demo` goes from nothing to a trained federation.
PY ?= python
SITES := hospital-a hospital-b hospital-c hospital-d hospital-e

.PHONY: install data certs tokens demo federation docker docker-federation test lint \
        experiments plots smoke flower he k8s-secrets clean

install:            ## virtualenv-free install into the current Python
	pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3"
	pip install -e ".[dev,experiments]"

data:               ## download DermaMNIST and split it into 5 non-IID hospitals (+ planted faults)
	fedmed prepare-data --out data/sites --sites 5 --alpha 0.5

certs:              ## federation CA, server + per-hospital certs, CRL, token signing key
	fedmed pki init --out certs --sites $(SITES) --extra-clients auditor
	$(MAKE) tokens

tokens:
	mkdir -p certs/tokens
	for s in $(SITES); do fedmed token --pki certs --site $$s > certs/tokens/$$s.jwt; done
	fedmed token --pki certs --site auditor --role observer > certs/tokens/auditor.jwt

demo: data certs federation   ## everything, locally, as separate processes over mTLS

federation:         ## coordinator + 5 hospital processes on localhost
	scripts/local_federation.sh

docker:
	docker build -f deploy/docker/Dockerfile -t fedmed:latest .

docker-federation: docker   ## coordinator + 5 hardened containers on isolated networks
	deploy/docker/run_federation.sh

test:
	pytest

lint:
	ruff check src tests experiments

smoke:
	$(PY) experiments/run_experiments.py --suite smoke --workers 3 --force

experiments:        ## every benchmark in RESULTS.md (~1.5 h on 16 CPU cores)
	$(PY) experiments/run_experiments.py --suite all
	$(PY) experiments/he_benchmark.py

plots:
	$(PY) experiments/make_plots.py

flower:             ## same training code driven by Flower's simulation engine
	cd flower_app && flwr run . --stream

k8s-secrets:        ## create the Secrets the Helm chart expects (kubectl context must be set)
	kubectl create secret generic fedmed-coordinator-pki --from-file=ca.crt=certs/ca.crt \
	  --from-file=crl.pem=certs/crl.pem --from-file=server.crt=certs/server.crt \
	  --from-file=server.key=certs/server.key --from-file=token_signing.pub=certs/token_signing.pub
	for s in $(SITES); do kubectl create secret generic $$s-pki --from-file=ca.crt=certs/ca.crt \
	  --from-file=client.crt=certs/$$s.crt --from-file=client.key=certs/$$s.key \
	  --from-file=token.jwt=certs/tokens/$$s.jwt; done

clean:
	rm -rf runs certs data/sites .pytest_cache .ruff_cache
