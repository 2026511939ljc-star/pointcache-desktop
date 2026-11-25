"""Process one job in the Python runtime appropriate for its cache format."""

import json
import sys
from pathlib import Path

from sequence_cache import run


if __name__ == '__main__':
    try:
        options = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
        print(json.dumps(run(options), ensure_ascii=False))
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
