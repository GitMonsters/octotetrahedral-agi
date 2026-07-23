#!/usr/bin/env bash
# =============================================================================
# OctoTetrahedral AGI – macOS Production Setup
# =============================================================================
# Supported: macOS 12.3+ on Apple Silicon (M1/M2/M3+)
# Usage:     bash scripts/setup_macos_production.sh [--skip-brew] [--port PORT]
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_NAME="com.octotetrahedral.agi"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
LAUNCH_AGENTS_SRC="${REPO_ROOT}/LaunchAgents/${APP_NAME}.plist"
LAUNCH_AGENTS_DST="${HOME}/Library/LaunchAgents/${APP_NAME}.plist"
LOG_DIR="${HOME}/Library/Logs/OctoTetrahedralAGI"
PORT="${PORT:-8000}"
PYTHON_MIN_VERSION="3.9"
MACOS_MIN_VERSION="12.3"

# Colour helpers
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
die()   { error "$*"; exit 1; }

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
SKIP_BREW=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-brew) SKIP_BREW=1 ;;
    --port=*) PORT="${1#*=}" ;;
    --port)
      [[ $# -ge 2 ]] || die "--port requires a value"
      PORT="$2"; shift ;;
  esac
  shift
done

# ---------------------------------------------------------------------------
# 1. Verify macOS version
# ---------------------------------------------------------------------------
check_macos_version() {
  info "Checking macOS version..."
  local os_version
  os_version=$(sw_vers -productVersion)
  local major minor
  major=$(echo "$os_version" | cut -d. -f1)
  minor=$(echo "$os_version" | cut -d. -f2)
  local required_major=12 required_minor=3

  if (( major < required_major )) || \
     ( (( major == required_major )) && (( minor < required_minor )) ); then
    die "macOS ${MACOS_MIN_VERSION}+ required; found ${os_version}"
  fi
  info "macOS ${os_version} ✓"
}

# ---------------------------------------------------------------------------
# 2. Verify Apple Silicon
# ---------------------------------------------------------------------------
check_apple_silicon() {
  info "Checking CPU architecture..."
  local arch
  arch=$(uname -m)
  if [[ "$arch" != "arm64" ]]; then
    warn "Apple Silicon (arm64) expected for Metal GPU acceleration; found ${arch}."
    warn "The service will fall back to CPU – performance will be lower."
  else
    info "Apple Silicon detected ✓"
  fi
}

# ---------------------------------------------------------------------------
# 3. Install Homebrew (optional)
# ---------------------------------------------------------------------------
install_homebrew() {
  if [[ "$SKIP_BREW" -eq 1 ]]; then
    warn "Homebrew installation skipped (--skip-brew)."
    return
  fi
  if command -v brew &>/dev/null; then
    info "Homebrew already installed ✓"
  else
    info "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add brew to PATH for the rest of this script
    eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || /usr/local/bin/brew shellenv 2>/dev/null)"
  fi
}

# ---------------------------------------------------------------------------
# 4. Install Python 3.9+
# ---------------------------------------------------------------------------
install_python() {
  info "Checking Python version..."
  local python_bin
  python_bin=$(command -v python3 || true)

  if [[ -n "$python_bin" ]]; then
    local ver
    ver=$("$python_bin" -c "import sys; print('%d.%d' % sys.version_info[:2])")
    local req_major=3 req_minor=9
    local maj min
    maj=$(echo "$ver" | cut -d. -f1)
    min=$(echo "$ver" | cut -d. -f2)
    if (( maj >= req_major )) && (( min >= req_minor )); then
      info "Python ${ver} ✓"
      return
    fi
  fi

  if command -v brew &>/dev/null; then
    info "Installing Python 3.11 via Homebrew..."
    brew install python@3.11
  else
    die "Python ${PYTHON_MIN_VERSION}+ not found and Homebrew unavailable. Install Python manually."
  fi
}

# ---------------------------------------------------------------------------
# 5. Create virtual environment
# ---------------------------------------------------------------------------
create_venv() {
  info "Setting up Python virtual environment at ${VENV_DIR}..."
  if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
    info "Virtual environment created ✓"
  else
    info "Virtual environment already exists ✓"
  fi
}

# ---------------------------------------------------------------------------
# 6. Install dependencies
# ---------------------------------------------------------------------------
install_dependencies() {
  info "Installing Python dependencies..."
  local pip="${VENV_DIR}/bin/pip"
  "$pip" install --upgrade pip --quiet

  # PyTorch with Metal (MPS) support – nightly has best MPS coverage
  info "Installing PyTorch with Metal backend..."
  "$pip" install --upgrade \
    torch torchvision torchaudio \
    --quiet

  # Project requirements
  if [[ -f "${REPO_ROOT}/requirements.txt" ]]; then
    "$pip" install -r "${REPO_ROOT}/requirements.txt" --quiet
  fi
  if [[ -f "${REPO_ROOT}/requirements-dev.txt" ]]; then
    "$pip" install -r "${REPO_ROOT}/requirements-dev.txt" --quiet
  fi
  info "Dependencies installed ✓"
}

# ---------------------------------------------------------------------------
# 7. Verify Metal backend
# ---------------------------------------------------------------------------
verify_metal() {
  info "Verifying Metal (MPS) backend..."
  local python="${VENV_DIR}/bin/python"
  if "$python" -c "
import torch, sys
if torch.backends.mps.is_available():
    t = torch.ones(2, 2, device='mps')
    assert t.sum().item() == 4.0
    print('Metal MPS backend verified ✓')
    sys.exit(0)
else:
    print('Metal MPS not available – will use CPU fallback')
    sys.exit(0)
" 2>&1; then
    :
  else
    warn "Metal backend check failed; CPU fallback will be used."
  fi
}

# ---------------------------------------------------------------------------
# 8. Set up logging directory
# ---------------------------------------------------------------------------
setup_logging() {
  info "Creating log directory at ${LOG_DIR}..."
  mkdir -p "${LOG_DIR}"
  info "Log directory ready ✓"
}

# ---------------------------------------------------------------------------
# 9. Install LaunchAgent (auto-start on login)
# ---------------------------------------------------------------------------
install_launch_agent() {
  info "Installing LaunchAgent..."

  if [[ ! -f "${LAUNCH_AGENTS_SRC}" ]]; then
    warn "LaunchAgent plist not found at ${LAUNCH_AGENTS_SRC}; skipping service installation."
    return
  fi

  mkdir -p "${HOME}/Library/LaunchAgents"

  # Patch working directory and log paths in a temp copy
  local tmp_plist
  tmp_plist=$(mktemp /tmp/com.octotetrahedral.plist.XXXXXX)
  sed \
    -e "s|__REPO_ROOT__|${REPO_ROOT}|g" \
    -e "s|__LOG_DIR__|${LOG_DIR}|g" \
    -e "s|__VENV_DIR__|${VENV_DIR}|g" \
    -e "s|__PORT__|${PORT}|g" \
    "${LAUNCH_AGENTS_SRC}" > "${tmp_plist}"

  cp "${tmp_plist}" "${LAUNCH_AGENTS_DST}"
  rm -f "${tmp_plist}"

  # Unload any previous version, then load
  launchctl unload "${LAUNCH_AGENTS_DST}" 2>/dev/null || true
  launchctl load -w "${LAUNCH_AGENTS_DST}"
  info "LaunchAgent installed and loaded ✓"
}

# ---------------------------------------------------------------------------
# 10. Configure firewall (application-level allow)
# ---------------------------------------------------------------------------
configure_firewall() {
  local python="${VENV_DIR}/bin/python"
  if command -v /usr/libexec/ApplicationFirewall/socketfilterfw &>/dev/null; then
    info "Configuring application firewall..."
    /usr/libexec/ApplicationFirewall/socketfilterfw --add "$python" 2>/dev/null || true
    /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp "$python" 2>/dev/null || true
    info "Firewall configured ✓"
  else
    warn "socketfilterfw not found; skipping firewall configuration."
  fi
}

# ---------------------------------------------------------------------------
# 11. Validate installation
# ---------------------------------------------------------------------------
validate_installation() {
  info "Validating installation..."
  local python="${VENV_DIR}/bin/python"

  "$python" -c "
import sys
sys.path.insert(0, '${REPO_ROOT}')
import production_config as cfg
print(f'Config loaded – ENV={cfg.ENV}, MODEL_VERSION={cfg.MODEL_VERSION}')
" || warn "production_config import check failed – review manually."

  info "Installation validation complete ✓"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
  echo "============================================================"
  echo " OctoTetrahedral AGI – macOS Production Setup"
  echo "============================================================"
  echo ""

  check_macos_version
  check_apple_silicon
  install_homebrew
  install_python
  create_venv
  install_dependencies
  verify_metal
  setup_logging
  install_launch_agent
  configure_firewall
  validate_installation

  echo ""
  echo "============================================================"
  echo " ✅  Setup complete!"
  echo ""
  echo "  Service:  ${APP_NAME}"
  echo "  Port:     ${PORT}"
  echo "  Logs:     ${LOG_DIR}"
  echo "  Venv:     ${VENV_DIR}"
  echo ""
  echo "  Start:    launchctl start ${APP_NAME}"
  echo "  Stop:     launchctl stop  ${APP_NAME}"
  echo "  Status:   launchctl list  ${APP_NAME}"
  echo "  Health:   bash scripts/health_check_macos.sh"
  echo "============================================================"
}

main "$@"
