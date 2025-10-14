#!/usr/bin/env bash
set -euo pipefail

trap 'err=$?; log_event "Desktop dev shell failed with exit code $err" "failure" "error"; exit $err' ERR

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
  printf '{"filename":"scripts/start_desktop.sh","timestamp":"%s","classname":"StartDesktop","function":"%s","system_section":"%s","line_num":%s,"error":%s,"db_phase":"none","method":"NONE","message":"%s"}\n' \
    "$timestamp" "$func_name" "$section" "$line_number" "$error_flag" "$escaped_message"
}

print_quality_prompt() {
  cat <<'PROMPT'
[Quality checklist]
* Could this change affect unexpected files/systems?
* Any hidden dependencies or cascades?
* What edge cases and failure modes are unhandled?
* If stuck, work backward from the desired outcome.
PROMPT
}

ensure_command() {
  local cmd="$1"
  local description="$2"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    log_event "Required command '$cmd' for $description not found on PATH" "deps" "error"
    cat >&2 <<ERR
Missing dependency: $cmd
Install $description and ensure it is available on your PATH before re-running this script.
ERR
    exit 1
  fi
}

ensure_cargo_tauri() {
  if ! cargo tauri --version >/dev/null 2>&1; then
    log_event "cargo-tauri CLI not available" "deps" "error"
    cat >&2 <<'ERR'
The cargo-tauri CLI is required to launch the desktop shell.
Install it via: cargo install tauri-cli
ERR
    exit 1
  fi
}

main() {
  print_quality_prompt

  local project_root
  project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  log_event "Project root detected at $project_root"

  cd "$project_root"

  local desktop_dir
  desktop_dir="${DESKTOP_APP_ROOT:-$project_root/apps/desktop}"
  local web_dir
  web_dir="${DESKTOP_WEB_DIR:-$project_root/apps/web}"

  log_event "Resolved desktop workspace to $desktop_dir" "paths"
  log_event "Resolved web workspace to $web_dir" "paths"

  if [[ ! -d "$desktop_dir" ]]; then
    log_event "Desktop workspace directory not found" "paths" "error"
    cat >&2 <<ERR
Expected a Tauri desktop workspace at:
  $desktop_dir
Set DESKTOP_APP_ROOT to override the path or initialize the desktop workspace (e.g. git submodule, adjacent repository checkout).
ERR
    exit 1
  fi

  local tauri_manifest="$desktop_dir/src-tauri/Cargo.toml"
  if [[ ! -f "$tauri_manifest" ]]; then
    log_event "Tauri Cargo manifest missing at $tauri_manifest" "paths" "error"
    cat >&2 <<ERR
Missing src-tauri Cargo manifest at:
  $tauri_manifest
Ensure the desktop workspace contains a valid Tauri project structure.
ERR
    exit 1
  fi

  if [[ ! -d "$web_dir" ]]; then
    log_event "Web workspace directory not found" "paths" "error"
    cat >&2 <<ERR
Expected a web frontend workspace at:
  $web_dir
Set DESKTOP_WEB_DIR to override the path or fetch the web assets (e.g. git submodule update or clone).
ERR
    exit 1
  fi

  local package_json="$web_dir/package.json"
  if [[ ! -f "$package_json" ]]; then
    log_event "package.json missing from web workspace" "paths" "error"
    cat >&2 <<ERR
package.json not found at:
  $package_json
Verify that the frontend dependencies have been checked out correctly.
ERR
    exit 1
  fi

  ensure_command npm "Node.js + npm"
  ensure_command cargo "Rust toolchain"
  ensure_cargo_tauri

  if [[ "${SKIP_NPM_INSTALL:-0}" == "1" ]]; then
    log_event "Skipping npm install because SKIP_NPM_INSTALL=1" "deps"
  else
    log_event "Installing frontend dependencies with npm" "deps"
    npm install --prefix "$web_dir"
  fi

  log_event "Launching cargo-tauri dev shell" "desktop-run"
  (cd "$desktop_dir" && cargo tauri dev "$@")

  log_event "cargo-tauri dev shell exited cleanly" "complete"
}

main "$@"
