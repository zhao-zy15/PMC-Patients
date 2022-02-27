"""
    Parse Mesh tree and extract Mesh terms of type Disease.
"""

import os
import json

# PMC_OA dataset downloaded from https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/
data_dir = "../../../../PMC_OA"
mtree = open(os.path.join(data_dir, "mtrees2022.bin"), "rb")
Mesh2tree = {}
for line in mtree:
    name, tree_id = line.strip().split(b';')
    if tree_id.startswith(b"C"):
        Mesh2tree[str(name)[2:-1]] = str(tree_id)[2:-1]

json.dump(Mesh2tree, open(os.path.join(data_dir, "Mesh2tree.json"), "w"), indent = 4)