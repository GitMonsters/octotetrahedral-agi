"""Allow ``python -m eval_harness <command>`` invocation."""
from eval_harness.cli import main
import sys

sys.exit(main())
