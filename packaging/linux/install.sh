#!/usr/bin/env bash
# Install MailAI as a systemd user service.
# Usage:
#   bash packaging/linux/install.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNIT_SRC="$REPO_ROOT/packaging/linux/mailai.service"
UNIT_DST="$HOME/.config/systemd/user/mailai.service"
TARGET_DIR="$HOME/mailai"

if [[ "$REPO_ROOT" != "$TARGET_DIR" ]]; then
  echo "Note: the service expects MailAI at $TARGET_DIR but you're installing from $REPO_ROOT."
  echo "Either move/symlink the repo to $TARGET_DIR or edit WorkingDirectory in the unit file."
fi

mkdir -p "$(dirname "$UNIT_DST")"
cp "$UNIT_SRC" "$UNIT_DST"

systemctl --user daemon-reload
systemctl --user enable mailai.service
systemctl --user start mailai.service

echo
echo "MailAI installed as a user service."
echo "  status:  systemctl --user status mailai"
echo "  logs:    journalctl --user -u mailai -f"
echo "  stop:    systemctl --user stop mailai"
echo "  remove:  systemctl --user disable --now mailai && rm $UNIT_DST"
