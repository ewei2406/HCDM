import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import Utils.Export as Export


res = {
    "col1": "123",
    "col3": "abc",
    "col": 123.23
}

Export.saveData("../../Results/Results.csv", res)