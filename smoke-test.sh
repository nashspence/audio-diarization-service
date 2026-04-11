#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

set -a
# shellcheck disable=SC1091
source .env
set +a

HOST_PORT="${HOST_PORT:-8000}"
READY_TIMEOUT="${SMOKE_READY_TIMEOUT_SECONDS:-1800}"
REQUEST_TIMEOUT="${SMOKE_REQUEST_TIMEOUT_SECONDS:-3600}"
SERVICE_NAME="diarizen"
RESPONSE_FILE="$(mktemp)"
READY_FILE="$(mktemp)"

resolve_host() {
  if getent hosts host.docker.internal >/dev/null 2>&1; then
    printf '%s' "host.docker.internal"
    return
  fi
  printf '%s' "127.0.0.1"
}

BASE_URL="http://$(resolve_host):${HOST_PORT}"
READY_URL="${BASE_URL}/health/ready"

cleanup() {
  docker compose down --remove-orphans >/dev/null 2>&1 || true
  rm -f "${RESPONSE_FILE}" "${READY_FILE}"
}

trap cleanup EXIT

docker compose down --remove-orphans >/dev/null 2>&1 || true
docker compose up -d --build

deadline=$((SECONDS + READY_TIMEOUT))
while true; do
  status_code="$(curl -sS -o "${READY_FILE}" -w '%{http_code}' "${READY_URL}" || true)"
  if [[ "${status_code}" == "200" ]]; then
    break
  fi
  if [[ "${status_code}" == "503" ]] && jq -e '.status == "error"' "${READY_FILE}" >/dev/null 2>&1; then
    echo "Service entered an unrecoverable startup error:"
    cat "${READY_FILE}"
    docker compose logs --no-color "${SERVICE_NAME}" || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for ${SERVICE_NAME} readiness."
    cat "${READY_FILE}" || true
    docker compose logs --no-color "${SERVICE_NAME}" || true
    exit 1
  fi
  sleep 10
done

container_id="$(docker compose ps -q "${SERVICE_NAME}")"
health_deadline=$((SECONDS + 90))
while true; do
  health_status="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "${container_id}")"
  if [[ "${health_status}" == "healthy" ]]; then
    break
  fi
  if (( SECONDS >= health_deadline )); then
    echo "Container never reported healthy after readiness:"
    cat "${READY_FILE}"
    docker compose logs --no-color "${SERVICE_NAME}" || true
    exit 1
  fi
  sleep 3
done

curl -fsS \
  --max-time "${REQUEST_TIMEOUT}" \
  -X POST "${BASE_URL}/v1/diarize" \
  -F "file=@${PROJECT_ROOT}/test.opus;type=audio/ogg" \
  >"${RESPONSE_FILE}"

jq -e '
  .model.repo_id == "BUT-FIT/diarizen-wavlm-large-s80-md-v2" and
  (.summary.speaker_count | type == "number") and
  (.summary.segment_count | type == "number") and
  (.summary.segment_count >= 1) and
  (.speakers | type == "array") and
  (.segments | type == "array") and
  (.segments | length >= 1) and
  all(.segments[]; (.start_seconds | type == "number") and (.end_seconds | type == "number") and (.end_seconds > .start_seconds) and (.speaker | type == "string"))
' "${RESPONSE_FILE}" >/dev/null

jq '{summary, speakers, first_segment: .segments[0]}' "${RESPONSE_FILE}"
