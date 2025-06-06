import numpy as np
import fileinput


def array_to_string(array):
    lines = []
    
    for row in array:
        line = "{:7}, {:12.9f}, {:12.9f}, {:12.9f}".format(int(row[0]), row[1], row[2], row[3])
        lines.append(line)

    result_string = "\n".join(lines)

    if not result_string.endswith('\n'):
        result_string += '\n'

    return result_string

def overwrite_nodes_inp(inp_file_path, node_array):
    with open(inp_file_path, 'r') as file:
        lines = file.readlines()
    
    node_string = array_to_string(node_array)

    with open(inp_file_path, 'w') as file:
        inside_node_block = False

        for line in lines:
            if line == '*Node\n':
                inside_node_block = True
                file.write(line)
                # Begin with *Node
                continue
            elif '*Element, type=STRI3' in line:
                inside_node_block = False

                file.write(node_string)  # Insert new content
                file.write(line)  # Write the *Element line
                continue
            
            if not inside_node_block:
                file.write(line)

def extract_nodes(inp_file_path):
    node_data = []

    with open(inp_file_path, 'r') as file:
        lines = file.readlines()
        
        # Flag to indicate we are in the nodes section
        in_nodes_section = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("*Node"):
                in_nodes_section = True  # We reached the nodes section
                continue
            
            # If we are in the nodes section, process the node data
            if in_nodes_section:
                # Check for the end of nodes section (which would be indicated by next section starting with *)
                if line.startswith("*"):
                    break
                
                # Process the node line
                if line:  # Ensure the line is not empty
                    parts = line.split(',')
                    node_id = int(parts[0])  # Node ID
                    coords = [float(coord) for coord in parts[1:]]  # Convert coordinates to float
                    # Add the node ID and coordinates to the list
                    node_data.append([node_id] + coords)

    # Convert the list to a NumPy array
    node_array = np.array(node_data)
    return node_array

def sort_nodes_xyz(nodes, ny, delta_y):
    node_sort = []

    for i in range(ny):
        # Get nodes of same y
        idx = np.argwhere((nodes[:,2] < (i*delta_y + delta_y/2)) * (nodes[:,2] > (i*delta_y - delta_y/2)))[:,0]
        node_rows = nodes[idx,:]

        # Sort along x
        idx2 = np.argsort(node_rows[:,1])
        node_sort.append(node_rows[idx2,:])

    nodes_sorted = np.vstack(node_sort)

    return nodes_sorted

def sort_along_col(nodes, col):
    idx = np.argsort(nodes[:,col]).astype(int)
    nodes_reordered = nodes[idx]
    return nodes_reordered

def write_beading_to_inp(inp_file, beading_img):


    nodes = extract_nodes(inp_file)
    y_coords = nodes[:,2]
    flag_y_nodes_greater_0 = y_coords > 1e-15
    delta_y = np.min(y_coords[flag_y_nodes_greater_0])

    y_min = np.min(y_coords)
    y_max = np.max(y_coords)
    n_elem_y = int(y_max / delta_y)
    ny = n_elem_y + 1
    nodes_sorted = sort_nodes_xyz(nodes, ny, delta_y)

    # Assign beading to nodes
    nodes_sorted[:,3] = beading_img.flatten()

    # Sort node mat along columne
    nodes_id_ordered = sort_along_col(nodes_sorted, col = 0)

    overwrite_nodes_inp(inp_file, nodes_id_ordered)



def replace_line(file, key, replacement, skip_lines = 1):
    counter = None

    for line in fileinput.input(file, inplace=True, backup=".bak"):
        stripped = line.strip()

        # if we are *not* in countdown mode, check for the marker
        if counter is None:
            print(line, end="")               # emit the current line unmodified
            if stripped == key:
                counter = skip_lines                   # start counting down from x
            continue

        # if we get here, counter is an integer > 0
        counter -= 1

        if counter == 0:
            # this is the x-th line after the marker → emit replacement
            print(replacement, end="\n")
            counter = None                    # reset, so we go back to normal mode
        else:
            # still counting down → emit the original line
            print(line, end="")

def get_node_id(file, f_pos):
    nodes = extract_nodes(file)
    dist = (nodes[:,1] - f_pos[0])**2 + (nodes[:,2] - f_pos[1])**2
    f_node_id = nodes[np.argmin(dist),0]
    return f_node_id


