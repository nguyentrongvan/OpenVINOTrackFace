import numpy as np
from shapely.geometry import LineString
from collections import Counter


def confirm_across_gate(gate_line, track_line):
    line1_coords = [(gate_line[0], gate_line[1]), (gate_line[2], gate_line[3])]
    line2_coords = [(track_line[0], track_line[1]), (track_line[2], track_line[3])]
    ls1 = LineString(line1_coords)
    ls2 = LineString(line2_coords)

    intersection = ls1.intersection(ls2)

    if intersection.is_empty:
        return False  # Không có giao điểm
    elif intersection.geom_type == 'Point':
        x, y = intersection.x, intersection.y
        return True
    elif intersection.geom_type == 'MultiPoint':
        points = list(intersection)
        x, y = points[0].x, points[0].y
        return False
    else:
        # Giao điểm là một đoạn thẳng hoặc tập hợp các điểm
        return False
    

def confirm_cross_direct(tid, directions, cross_line_up, cross_line_down):
    direction = Counter(directions).most_common(1)[0][0]
    if direction == -1:
        if tid not in cross_line_up:
            cross_line_up.append(tid)  
            # if tid in cross_line_down:
            #     cross_line_down.remove(tid)
    elif direction == 1:
        if tid not in cross_line_down:
            cross_line_down.append(tid)
    return cross_line_up, cross_line_down
