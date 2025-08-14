import numpy as np

def generate_equidistant_points_on_line(p1, p2, distance): #p1,p2 list, distance float, return ndarray
    p1 = np.array(p1)
    p2 = np.array(p2)

    # calculate the direction vector of the line segment
    direction = p2 - p1
    
    line_length = np.linalg.norm(direction)
    
    # calculate the number of points to be generated
    num_points = int(np.ceil(line_length / distance)) + 1

    # generate points along the direction of the line segment
    t = np.linspace(0, 1, num_points)
    points = p1 + t[:, None] * direction
    
    return points


def get_rb_points_from_txt(file_dir,sample_distance): # return ndarray
    red_points = np.array([])
    blue_points = np.array([])
    all_points = []
    with open(file_dir, "r") as f:  
        point_num = int(f.readline())
        for i in range(point_num):
            point = [float(i) for i in f.readline().split()]
            all_points.append(point)
        
        f.readline()
        red_id = [int(i) for i in f.readline().split()] # red points id
        red_lines_num = int(f.readline())
        for i in range(red_lines_num):
            red_line = [int(i) for i in f.readline().split()]
            line_points = generate_equidistant_points_on_line(all_points[red_line[0]],all_points[red_line[1]],sample_distance)
            if len(red_points) == 0:
                red_points = line_points
            else :
                red_points = np.append(red_points,line_points,axis=0)

        f.readline()
        blue_id = [int(i) for i in f.readline().split()] # blue points id
        blue_lines_num = int(f.readline())
        for i in range(blue_lines_num):
            blue_line = [int(i) for i in f.readline().split()]
            line_points = generate_equidistant_points_on_line(all_points[blue_line[0]],all_points[blue_line[1]],sample_distance)
            if len(blue_points) == 0:
                blue_points = line_points
            else :
                blue_points = np.append(blue_points,line_points,axis=0)
        return red_points,blue_points