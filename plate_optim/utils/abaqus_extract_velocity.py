# Basic Python libraries ===============================================================================================
import os
import sys
import numpy as np

# Abaqus imports =======================================================================================================
from odbAccess import *


def replace_extension(file_path, old_ext, new_ext):
    # Separate the file path into the base and the extension
    base, ext = os.path.splitext(file_path)
    
    # Check if the current extension matches the old one we want to replace
    if ext.lower() == old_ext.lower():
        # Return the new file path with the new extension
        return base + "." + new_ext
    else:
        # If the extension doesn't match, return the original path
        return file_path

def main(inp_file):

    # Read the node values from the odb file ===============================================================================

    odb_file = replace_extension(inp_file, ".inp", "odb")

    odb = openOdb(path=odb_file)
    instances = odb.rootAssembly.instances

    step_keys = odb.steps.keys()
    stp_ = odb.steps[step_keys[-1]]
    inst_keys = instances.keys()

    set_ = instances[inst_keys[-1]].nodeSets['ALLNODES']
    frf = np.zeros((np.size(stp_.frames), 4))
    for ffi, frame in enumerate(stp_.frames):
        val = np.asarray([v.data + 1j * v.conjugateData for v in frame.fieldOutputs['V'].getSubset(region=set_).values])
        for kk, v in enumerate(val.T):
            frf[ffi, kk + 1] = np.mean(abs(v)**2)
        frf[ffi, 0] = frame.frequency
    np.savetxt('Velocity_{}.txt'.format('AllNodes'), frf)

if __name__ == "__main__":
    
    inp_file_path = sys.argv[-1]

    base_folder = os.path.dirname(inp_file_path)
    file_name = os.path.basename(inp_file_path)

    os.chdir(base_folder)

    main(file_name)


