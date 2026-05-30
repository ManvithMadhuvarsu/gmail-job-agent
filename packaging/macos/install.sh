#!/usr/bin/env bash
# Install MailAI as a launchd user agent (loads at login).
# Usage:
#   bash packaging/macos/install.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PLIST_TEMPLATE="$REPO_ROOT/packaging/macos/com.mailai.daemon.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.mailai.daemon.plist"

mkdir -p "$(dirname "$PLIST_DST")"
sed "s|__HOME__|$HOME|g" "$PLIST_TEMPLATE" > "$PLIST_DST"

launchctl unload "$PLIST_DST" 2>/dev/null || true
launchctl load -w "$PLIST_DST"

echo
echo "MailAI installed as a launchd user agent."
echo "  status:  launchctl list | grep com.mailai.daemon"
echo "  logs:    tail -f $HOME/mailai/data/daemon.log"
echo "  stop:    launchctl unload $PLIST_DST"
echo "  remove:  launchctl unload $PLIST_DST && rm $PLIST_DST"
