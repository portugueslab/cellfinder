import numpy as np


def cal_eucl_dist3d(cell0, cell1, real_units):
    # real units in order: z, y, x
    x = [cell0.z, cell0.y, cell0.x]
    y = [cell1.z, cell1.y, cell1.x]
    z_dist = (x[0] - y[0]) * real_units[0]
    y_dist = (x[1] - y[1]) * real_units[1]
    x_dist = (x[2] - y[2]) * real_units[2]
    return abs(np.sqrt(z_dist**2 + y_dist**2 + x_dist**2))


def mediate_coord(input_list):
    n_coord = len(input_list)
    output = [0, 0, 0]
    for coord in input_list:
        output = [np.sum(x) for x in zip(output, coord)]
    output = [number / n_coord for number in output]
    return output


def proximity_filter(input_list, max_dist, real_units):
    cells = []
    rois = input_list.copy()
    while rois:
        quest_parent = [rois.pop(0)]
        cell_container = [[quest_parent[0].z, quest_parent[0].y, quest_parent[0].x]]
        while quest_parent:
            parent = quest_parent.pop(0)
            for n, roi in enumerate(rois):
                dist = cal_eucl_dist3d(parent, roi, real_units)
                if dist <= max_dist:
                    cell_container.append([roi.z, roi.y, roi.x])
                    quest_parent.append(roi)
                    rois.pop(n)

        cells.append(mediate_coord(cell_container))
    return cells

