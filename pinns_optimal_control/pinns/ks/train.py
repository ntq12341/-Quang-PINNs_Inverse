from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pinns_optimal_control.pinns.ks.optimal_control_3_3_2 import main


if __name__ == "__main__":
    main()
