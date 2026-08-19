#!/usr/bin/env bash
# CL-01 / RL-040: write a starting capital declaration for the perp and spot bots.
#
# `capital_declaration.py` has NO DEFAULT ON PURPOSE - a missing or malformed file
# stops the bots opening rather than falling back to numbers nobody chose. That
# makes deployment order matter: the file must exist BEFORE the code that demands
# it is deployed, or there is a window where a live bot cannot trade for a reason
# that looks like a crash. This script closes that window.
#
# It REFUSES TO OVERWRITE an existing declaration. The file is the user's own
# decision about how much money the system may spend, and a script that silently
# replaced it would be the one thing this whole subsystem exists to prevent. Pass
# --force to replace it deliberately.
#
# Every number below is a PLACEHOLDER chosen to be conservative and to match what
# the bots are measured against today (two bots x 1,000 USDT declared bankroll).
# Edit the file, do not edit this script - the bots re-read the file on the next
# poll without a restart.
set -uo pipefail

CAPTURE_ROOT=${CAPTURE_ROOT:-$HOME/capture}
TARGET="$CAPTURE_ROOT/segment/capital.json"
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

if [ -e "$TARGET" ] && [ "$FORCE" -eq 0 ]; then
    echo "refusing to overwrite the existing declaration at $TARGET" >&2
    echo "it is a decision about how much capital the bots may spend." >&2
    echo "edit it directly, or pass --force to replace it." >&2
    exit 1
fi

mkdir -p "$(dirname "$TARGET")"

cat > "$TARGET" <<JSON
{
  "declared_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",

  "portfolio_usdt": "2000",
  "per_bot_cap_fraction": "0.60",

  "min_margin_per_trade_usdt": "5",
  "max_margin_per_trade_usdt": "50",

  "leverage": {
    "rule": "volatility_targeted",
    "floor": "1",
    "ceiling": { "perp": "20", "spot": "5" },
    "target_annual_vol_pct": "40"
  },

  "spot_borrow_annual_pct": "8.0",
  "maintenance_margin_rate": "0.005"
}
JSON

echo "wrote $TARGET"

# Validate what was just written through the same parser the bots use. A starting
# file this script emitted but the code rejects would be the worst possible state
# to hand over, and it is cheap to rule out here.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHONPATH="$REPO/src" "${SEGMENT_PYTHON:-$REPO/.venv/bin/python}" - "$TARGET" <<'PY'
import json, sys
from segment.capital_declaration import parse_declaration, DeclarationRejected

path = sys.argv[1]
try:
    declaration = parse_declaration(json.loads(open(path).read()))
except (DeclarationRejected, ValueError) as error:
    print(f"REJECTED by the parser the bots use: {error}", file=sys.stderr)
    raise SystemExit(2)

print(f"  portfolio        {declaration.portfolio_usdt} USDT")
print(f"  per-bot cap      {declaration.cap_for('perp')} USDT "
      f"({declaration.per_bot_cap_fraction} of the portfolio)")
print(f"  margin per trade {declaration.min_margin_per_trade_usdt} "
      f"to {declaration.max_margin_per_trade_usdt} USDT")
print(f"  leverage rule    {declaration.leverage_rule}, "
      f"perp up to {declaration.ceiling_for('perp')}x, "
      f"spot up to {declaration.ceiling_for('spot')}x")
print(f"  spot borrow      {declaration.spot_borrow_annual_pct}% annual")
PY
