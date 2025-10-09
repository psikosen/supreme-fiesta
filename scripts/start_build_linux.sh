#!/usr/bin/env bash
set -euo pipefail

trap 'err=$?; line=${BASH_LINENO[0]:-0}; log_event "Start build failed with exit code $err" "failure" "error"; exit $err' ERR

json_escape() {
  local str="$1"
  str=${str//\\/\\\\}
  str=${str//\"/\\\"}
  str=${str//$'\n'/\\n}
  printf '%s' "$str"
}

log_event() {
  local message="$1"
  local section="${2:-setup}"
  local level="${3:-info}"
  local error_flag=false
  if [[ "$level" == "error" ]]; then
    error_flag=true
  fi
  local timestamp
  timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  local func_name="${FUNCNAME[1]:-main}"
  local line_number="${BASH_LINENO[0]:-0}"
  local escaped_message
  escaped_message=$(json_escape "$message")
  printf '{"filename":"scripts/start_build_linux.sh","timestamp":"%s","classname":"StartBuildLinux","function":"%s","system_section":"%s","line_num":%s,"error":%s,"db_phase":"none","method":"NONE","message":"%s"}\n' \
    "$timestamp" "$func_name" "$section" "$line_number" "$error_flag" "$escaped_message"
}

print_sherlock_prompt() {
  cat <<'PROMPT'
[Continuous skepticism (Sherlock Protocol)]
* Could this change affect unexpected files/systems?
* Any hidden dependencies or cascades?
* What edge cases and failure modes are unhandled?
* If stuck, work backward from the desired outcome.
PROMPT
}

ensure_system_dependencies() {
  if [[ "${SKIP_SYSTEM_DEPS:-0}" == "1" ]]; then
    log_event "Skipping system dependency installation because SKIP_SYSTEM_DEPS=1" "system-deps"
    return
  fi

  if command -v apt-get >/dev/null 2>&1; then
    log_event "Ensuring PortAudio and build tooling via apt" "system-deps"
    local packages=(portaudio19-dev python3-dev build-essential)
    local needs_update=1
    for pkg in "${packages[@]}"; do
      if dpkg -s "$pkg" >/dev/null 2>&1; then
        log_event "APT package '$pkg' already installed" "system-deps"
      else
        if (( needs_update )); then
          if (( EUID == 0 )); then
            apt-get update
          elif command -v sudo >/dev/null 2>&1; then
            sudo apt-get update
          else
            log_event "Cannot update apt repositories without root privileges" "system-deps" "error"
            echo "Install $pkg manually or rerun with sudo." >&2
            exit 1
          fi
          needs_update=0
        fi

        if (( EUID == 0 )); then
          apt-get install -y "$pkg"
        elif command -v sudo >/dev/null 2>&1; then
          sudo apt-get install -y "$pkg"
        else
          log_event "Cannot install '$pkg' without root privileges" "system-deps" "error"
          echo "Install $pkg manually or rerun with sudo." >&2
          exit 1
        fi
      fi
    done
  else
    log_event "apt-get not available; ensure PortAudio headers are installed manually" "system-deps"
  fi
}

main() {
  print_sherlock_prompt

  local project_root
  project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  log_event "Project root detected at $project_root" "setup"

  cd "$project_root"

  ensure_system_dependencies

  local python_bin
  python_bin="${PYTHON:-python3.11}"
  if ! command -v "$python_bin" >/dev/null 2>&1; then
    log_event "Preferred python bin '$python_bin' not found; falling back to python3" "python"
    if command -v python3 >/dev/null 2>&1; then
      python_bin=python3
    else
      log_event "Python 3 interpreter not found on PATH" "python" "error"
      echo "Python 3.11+ is required." >&2
      exit 1
    fi
  fi
  log_event "Using Python interpreter: $(command -v "$python_bin")" "python"

  log_event "Running bootstrap sequence" "bootstrap"
  PYTHON="$python_bin" "$project_root/scripts/bootstrap.sh"

  local install_args=(--with dev)
  IFS=' ' read -r -a extras <<< "${POETRY_EXTRAS:-tts asr audio}"
  for extra in "${extras[@]}"; do
    if [[ -n "$extra" ]]; then
      install_args+=(--extras "$extra")
    fi
  done

  log_event "Installing project dependencies with extras: ${extras[*]:-none}" "deps"
  poetry env use "$python_bin"
  poetry install "${install_args[@]}"

  log_event "Running pytest suite" "tests"
  poetry run pytest

  log_event "Running Ruff lint checks" "tests"
  poetry run ruff check .

  log_event "Building Poetry distribution artifacts" "build"
  poetry build

  log_event "Start build workflow completed successfully" "complete"
}

main "$@"
