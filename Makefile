.PHONY: build up down test mock check e1 e2 e3 e4 e5 plots fit
build:  ; docker compose -f docker/docker-compose.yml build
up:     ; docker compose -f docker/docker-compose.yml up -d
down:   ; docker compose -f docker/docker-compose.yml down
test:   ; python -m unittest discover -s tests -v
mock:   ; EDGESPLIT_MOCK=1 python -m edgesplit.experiments e2 && EDGESPLIT_MOCK=1 python -m edgesplit.plots
check e1 e2 e3 e4 e5: ; python -m edgesplit.experiments $@
fit:    ; python -m edgesplit.analysis fit
plots:  ; python -m edgesplit.plots
