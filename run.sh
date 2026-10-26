#!/usr/bin/env bash

set -u

cd "$(dirname "$0")"

API_PORT=8000
WEB_PORT=5173
VENV=.venv
PY="$VENV/bin/python"

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
warn() { printf '\033[33m%s\033[0m\n' "$1"; }
fail() { printf '\033[31m%s\033[0m\n' "$1" >&2; exit 1; }

free_port() {
    local port=$1
    local pids
    pids=$(lsof -ti:"$port" 2>/dev/null) || true
    if [ -n "$pids" ]; then
        echo "  freeing port $port"
        # shellcheck disable=SC2086
        kill -9 $pids 2>/dev/null || true
        sleep 1
    fi
}

setup() {
    command -v python3 >/dev/null || fail "python3 is not installed"
    command -v cc >/dev/null || command -v clang >/dev/null || command -v gcc >/dev/null \
        || fail "no C compiler found -- install Xcode command line tools: xcode-select --install"

    if [ ! -x "$PY" ]; then
        echo "  creating the virtualenv"
        python3 -m venv "$VENV" || fail "could not create $VENV"
    fi
    echo "  installing Python dependencies"
    "$VENV/bin/pip" install -q -r requirements.txt || fail "pip install failed"

    if [ ! -d frontend/node_modules ]; then
        command -v npm >/dev/null || fail "npm is not installed -- install Node.js"
        echo "  installing frontend dependencies (first run only)"
        (cd frontend && npm install --silent) || fail "npm install failed"
    fi
}

case "${1:-start}" in
    setup)
        bold "Setting up"
        setup
        echo
        bold "Done. Run ./run.sh to start it."
        exit 0
        ;;
    test)
        [ -x "$PY" ] || setup
        exec "$PY" -m pytest tests -q
        ;;
    stop)
        bold "Stopping"
        free_port "$API_PORT"
        free_port "$WEB_PORT"
        echo "  stopped"
        exit 0
        ;;
    start) ;;
    *) fail "unknown command: $1 (try: setup, test, stop, or no argument)" ;;
esac

bold "NLP-Driven Compiler Assistant"
echo

[ -x "$PY" ] && [ -d frontend/node_modules ] || setup

free_port "$API_PORT"
free_port "$WEB_PORT"

MODEL=$("$PY" -m nlpcc models 2>/dev/null | tr -d '\n' | sed 's/^ *//;s/ *$//')
if [ -z "$MODEL" ]; then
    MODEL="none"
fi
case "$MODEL" in
    *"no model available"*)
        warn "  No model: AI explanations, AI analysis and code generation are off."
        warn "  Everything else works. Configure your local API in .env:"
        warn "      NLPCC_LOCAL_URL=https://your-local-llm/v1"
        warn "      NLPCC_LOCAL_API_KEY=your-key"
        ;;
    *) echo "  model: $MODEL" ;;
esac
echo

cleanup() {
    echo
    echo "  stopping"
    [ -n "${API_PID:-}" ] && kill "$API_PID" 2>/dev/null
    [ -n "${WEB_PID:-}" ] && kill "$WEB_PID" 2>/dev/null
    wait 2>/dev/null
    exit 0
}
trap cleanup INT TERM

"$PY" -u -m nlpcc serve --port "$API_PORT" >/tmp/nlpcc-api.log 2>&1 &
API_PID=$!

(cd frontend && npm run dev -- --port "$WEB_PORT" >/tmp/nlpcc-web.log 2>&1) &
WEB_PID=$!

for _ in $(seq 1 40); do
    api_up=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$API_PORT/api/health" 2>/dev/null || echo 000)
    web_up=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$WEB_PORT/" 2>/dev/null || echo 000)
    [ "$api_up" = "200" ] && [ "$web_up" = "200" ] && break
    sleep 0.5
done

if [ "$api_up" != "200" ]; then
    warn "  the API did not start -- see /tmp/nlpcc-api.log"
    tail -5 /tmp/nlpcc-api.log 2>/dev/null
    cleanup
fi
if [ "$web_up" != "200" ]; then
    warn "  the web interface did not start -- see /tmp/nlpcc-web.log"
    tail -5 /tmp/nlpcc-web.log 2>/dev/null
    cleanup
fi

bold "  Open  http://localhost:$WEB_PORT"
echo "  API   http://localhost:$API_PORT"
echo
echo "  Ctrl-C to stop both."
echo

command -v open >/dev/null && open "http://localhost:$WEB_PORT" 2>/dev/null

wait
