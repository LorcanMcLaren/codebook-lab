#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

OLLAMA_STARTED_BY_SCRIPT=0
SERVER_PID=""

cleanup() {
  if [[ "$OLLAMA_STARTED_BY_SCRIPT" == "1" && -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama is required but was not found in PATH."
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but was not found in PATH."
  exit 1
fi

mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/outputs/metrics"

if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "Starting Ollama server locally..."
  nohup ollama serve > "$REPO_ROOT/logs/ollama.log" 2>&1 &
  SERVER_PID=$!
  OLLAMA_STARTED_BY_SCRIPT=1
  sleep 5
fi

if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "Ollama server is not reachable on http://127.0.0.1:11434"
  exit 1
fi

TASKS=($(python3 -c 'import yaml; p=yaml.safe_load(open("param_grid.yaml")); print(" ".join(map(str, p.get("tasks") or ["policy-sentiment"])))'))
MODELS=($(python3 -c 'import yaml; p=yaml.safe_load(open("param_grid.yaml")); print(" ".join(map(str, p.get("models") or ["gemma3:270m"])))'))
COUNTRY_ISO_CODE=$(python3 -c 'import yaml; p=yaml.safe_load(open("param_grid.yaml")); print(str(p.get("country_iso_code") or "USA"))')
USE_EXAMPLES=($(python3 -c 'import yaml; p=yaml.safe_load(open("param_grid.yaml")); print(" ".join(map(str, p.get("use_examples") or ["false"])))'))
PROMPT_TYPES=($(python3 -c 'import yaml; p=yaml.safe_load(open("param_grid.yaml")); print(" ".join(map(str, p.get("prompt_types") or ["standard"])))'))
TEMPERATURES=($(python3 -c 'import yaml; p=yaml.safe_load(open("param_grid.yaml")); print(" ".join(map(str, p.get("temperatures") or ["None"])))'))
TOP_PS=($(python3 -c 'import yaml; p=yaml.safe_load(open("param_grid.yaml")); print(" ".join(map(str, p.get("top_ps") or ["None"])))'))
PROCESS_TEXTBOXES=($(python3 -c 'import yaml; p=yaml.safe_load(open("param_grid.yaml")); print(" ".join(map(str, p.get("process_textboxes") or ["false"])))'))

for TASK in "${TASKS[@]}"; do
  INPUT="tasks/${TASK}/sample.csv"
  CODEBOOK="tasks/${TASK}/codebook.json"
  GROUND_TRUTH="tasks/${TASK}/ground-truth.csv"
  LABEL="$TASK"
  METRICS_OUTPUT="outputs/metrics/${TASK}_metrics_log.csv"

  for MODEL in "${MODELS[@]}"; do
    for USE_EXAMPLE in "${USE_EXAMPLES[@]}"; do
      for PROMPT_TYPE in "${PROMPT_TYPES[@]}"; do
        for TEMP in "${TEMPERATURES[@]}"; do
          for TOP_P in "${TOP_PS[@]}"; do
            for PROCESS_TEXTBOX in "${PROCESS_TEXTBOXES[@]}"; do
              echo "Pulling model: $MODEL"
              ollama pull "$MODEL"

              DATE=$(date +%Y-%m-%d_%H-%M-%S)
              MODEL_SAFE=${MODEL//:/-}
              EXPERIMENT_DIR="outputs/${TASK}/${MODEL_SAFE}_examples${USE_EXAMPLE}_${PROMPT_TYPE}"
              MODEL_ID="${MODEL}_examples${USE_EXAMPLE}_${PROMPT_TYPE}"

              if [[ "$TEMP" != "None" ]]; then
                EXPERIMENT_DIR="${EXPERIMENT_DIR}_temp${TEMP}"
                MODEL_ID="${MODEL_ID}_temp${TEMP}"
              fi

              if [[ "$TOP_P" != "None" ]]; then
                EXPERIMENT_DIR="${EXPERIMENT_DIR}_topp${TOP_P}"
                MODEL_ID="${MODEL_ID}_topp${TOP_P}"
              fi

              if [[ "$PROCESS_TEXTBOX" == "true" ]]; then
                EXPERIMENT_DIR="${EXPERIMENT_DIR}_textbox"
                MODEL_ID="${MODEL_ID}_textbox"
              fi

              EXPERIMENT_DIR="${EXPERIMENT_DIR}_${DATE}"
              mkdir -p "$EXPERIMENT_DIR"

              CSV_OUTPUT="${EXPERIMENT_DIR}/output.csv"
              REPORT_FILE="${EXPERIMENT_DIR}/classification_reports.txt"
              EMISSIONS_FILE="${EXPERIMENT_DIR}/emissions.csv"

              python3 ./pipeline/annotate.py \
                "$MODEL" \
                "$INPUT" \
                "$CODEBOOK" \
                "$CSV_OUTPUT" \
                "$EXPERIMENT_DIR" \
                --use_examples "$USE_EXAMPLE" \
                --prompt_type "$PROMPT_TYPE" \
                --temperature "$TEMP" \
                --top_p "$TOP_P" \
                --process_textbox "$PROCESS_TEXTBOX" \
                --country_iso_code "$COUNTRY_ISO_CODE"

              python3 ./pipeline/metrics.py \
                "$GROUND_TRUTH" \
                "$CSV_OUTPUT" \
                --timestamp "$DATE" \
                --label "$LABEL" \
                --output_csv "$METRICS_OUTPUT" \
                --model_id "$MODEL_ID" \
                --temperature "$TEMP" \
                --prompt_type "$PROMPT_TYPE" \
                --use_examples "$USE_EXAMPLE" \
                --top_p "$TOP_P" \
                --process_textbox "$PROCESS_TEXTBOX" \
                --codebook_path "$CODEBOOK" \
                --report_file "$REPORT_FILE" \
                --emissions_file "$EMISSIONS_FILE" \
                --experiment_directory "$EXPERIMENT_DIR" \
                --timing_file "$EXPERIMENT_DIR/timing_data.json" \
                --char_counts_file "$EXPERIMENT_DIR/char_counts.json"
            done
          done
        done
      done
    done
  done
done
