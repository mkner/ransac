# 
# ransacmod
#
# ransac module for ransac algorithm examples
# for planar surface  generation
#
# mk v0.05
#

import os

import numpy as np
import cv2

from matplotlib import pyplot as plt


class RoadViews:

    def __init__(self):
        # Define number of frames
        self.num_frames = 3

        # paths
        root_dir_path = os.path.dirname(os.path.realpath(__file__))
        self.image_dir = os.path.join(root_dir_path, 'image_data/rgb')
        self.depth_dir = os.path.join(root_dir_path, 'image_data/depth')
        self.segmentation_dir = os.path.join(
            root_dir_path, 'image_data/segmentation')

        # frame select
        self.current_frame = 0

        # image data
        self.image = None
        self.depth = None
        self.segmentation = None
        # other
        self.object_detection = None
        self.lane_midpoint = None

        self.k = np.array([[640, 0, 640],
                           [0, 640, 480],
                           [0, 0, 1]])
        
        # segmentation color mapping
        self.semseg_color_map = {
            'None': (0, 0, 127),  # Navy Blue
            'Buildings': (255, 0, 0),  # Red
            'Fences': (0, 0, 127),  # 
            'Other': (0, 0, 127),  # Navy Blue
            'Pedestrians': (0, 255, 255),  # Green
            'Poles': (255, 255, 255),  # White
            'RoadLines': (255, 0, 255),  # Purple
            'Roads': (0, 0, 255),  # Blue
            'Sidewalks': (255, 255, 0),  # Yellow
            'Vegetation': (0, 127, 0),  # 
            'Vehicles': (0, 255, 0),  # Teal
            'Walls': (0, 0, 127),  # like walls
            'Cyclist' : (0,127,200) #
        }
     
        # init by loading default frame in
        self.read_frame()


    def _read_image(self):
        im_name = self.image_dir + '/' + str(self.current_frame) + '.png'
        self.image = cv2.imread(im_name)[:, :, ::-1]

    def _read_depth(self):
        depth_name = self.depth_dir + '/' + str(self.current_frame) + '.dat'
        depth = np.loadtxt(
            depth_name,
            delimiter=',',
            dtype=np.float64) * 1000.0
        self.depth = depth

    def _read_segmentation(self):
        seg_name = self.segmentation_dir + '/' + \
            str(self.current_frame) + '.dat'
        self.segmentation = np.loadtxt(seg_name, delimiter=',')

    def _read_lane_midpoint(self):
        midpoint_dict = {0: np.array([800, 900]),
                         1: np.array([800, 900]),
                         2: np.array([700, 900])}
        
        self.lane_midpoint = midpoint_dict[self.current_frame]
        
        
    def xy_from_depth(self, depth): # K):
        
        """
        mk - from v0.23
        
        calc (x,y) coordinates (in camera frame) for each pixel in the image
        using the depth map for z coords and K intrinsic (calibration) matrix
    
        Arguments:
        depth -- array (H, W), contains a depth value (in meters) for image pixels
    
        Returns:
        X -- array (H, W) x coordinates in the camera coord frame for each pixel 
        Y -- array (H, W) y coordinates in the camera coord frame for each pixel 
        """
        #  caps in notation for matrices & 2d arrays 
        # lower case for vectors & scalars
    
        #vectorized version
     
        K=self.k
        
        Z = depth
        h,w = Z.shape
        
        cu = K[0,2]
        cv = K[1,2]
        
        #f = K[0,0] #focal length
        # note  fx & fy not always the same
        fx = K[0,0]
        fy = K[1,1]
        
        # construct [H x W] array grids of image coordinates (u,v)
        
        u = np.linspace(1,w,w) # cols (width)
        v = np.linspace(1,h,h) # rows (height)
        U,V = np.meshgrid(u,v)
        
        X = ((U-cu) * Z)/fx
        Y = ((V-cv) * Z)/fy
  
        return X, Y



    def read_frame(self):
        self._read_image()
        self._read_depth()
        self._read_segmentation()
        ##self._read_object_detection()
        self._read_lane_midpoint()
        

    def get_next_frame(self):
        self.current_frame += 1

        if self.current_frame > self.num_frames - 1:
            self.current_frame = self.num_frames - 1
            return False
        else:
            self.read_frame()
            return True


    def get_previous_frame(self):
        self.current_frame -= 1

        if self.current_frame < 0:
            self.current_frame = 0
            return False
        else:
            self.read_frame()
            return True


    def set_frame(self, frame_number):
        self.current_frame = frame_number
        if self.current_frame > 2:
            self.current_frame = 2
            self.read_frame()
        elif self.current_frame < 0:
            self.current_frame = 0
            self.read_frame()
        else:
            self.read_frame()


    def visualize_segmentation(self, segmented_image):
        #colored_segmentation = np.zeros(self.image.shape)
        colored_segmentation = np.ones(self.image.shape)
        colored_segmentation = colored_segmentation*self.semseg_color_map['None']
        colored_segmentation[segmented_image ==
                             1] = self.semseg_color_map['Buildings']
        colored_segmentation[segmented_image ==
                             4] = self.semseg_color_map['Pedestrians']
        colored_segmentation[segmented_image ==
                             5] = self.semseg_color_map['Poles']
        colored_segmentation[segmented_image ==
                             6] = self.semseg_color_map['RoadLines']
        colored_segmentation[segmented_image ==
                             7] = self.semseg_color_map['Roads']
        colored_segmentation[segmented_image ==
                             8] = self.semseg_color_map['Sidewalks']
        colored_segmentation[segmented_image ==
                             10] = self.semseg_color_map['Vehicles']
                
        return colored_segmentation.astype(np.uint8)



    def visualize_lanes(self, lane_lines):
        image_out = self.image
        for line in lane_lines:
            x1, y1, x2, y2 = line.astype(int)
    
            image_out = cv2.line(
                image_out.astype(
                    np.uint8), (x1, y1), (x2, y2), (255, 0, 255), 7)
    
        return image_out
    

    def show_overhead_view(self, segmentation):
        
        # top down view of mapped surface 
        
        depth = self.depth

        sz = depth.shape
        f = self.k[0, 0]
        c_u = self.k[0, 2]

        # generate a coordinate grid  
        # of the dimensions of the depth map
        
        u, v = np.meshgrid(np.arange(1, sz[1] + 1, 1),
                           np.arange(1, sz[0] + 1, 1))

        # calc x,y coordinates from 
        # K insrinsic matrix parameters
        
        xx = ((u - c_u) * depth) / f

        xx = xx * 10 + 200
        xx = np.maximum(0, np.minimum(xx, 399))

        depth = depth * 10
        depth[depth > 300] = np.nan

        occ_grid = np.full([301, 401], 0.5)

        for x, z, seg in zip(xx.flatten('C'), depth.flatten('C'),
                             segmentation.flatten('C')):
            if not(seg == 1):
                if not np.isnan(x) and not np.isnan(z):
                    x = int(x)
                    z = int(z)
                    occ_grid[z, x] = 1

        for x, z, seg in zip(xx.flatten('C'), depth.flatten('C'),
                             segmentation.flatten('C')):
            if seg == 1:
                if not np.isnan(x) and not np.isnan(z):
                    x = int(x)
                    z = int(z)
                    if not occ_grid[z, x] == 1:
                        occ_grid[z, x] = 0

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(occ_grid, cmap='Greys')

        labels = ax.get_xticks()
        labels = [str((label - 200) / 10.0) for label in labels]
        ax.set_xticklabels(labels)

        labels = ax.get_yticks()
        labels = [str(label / 10.0) for label in labels]
        ax.set_yticklabels(labels)

        ax.invert_yaxis()
        plt.show()


def calc_plane(xyz):
    """
    calulate plane coefficients a,b,c,d of the plane 
    generated by  ax+by+cz+d = 0

    Arguments:
    xyz -- array (3, N), contains points to generate plane equation
    k -- array  (3x3) intrinsic camera matrix

    Returns:
    p -- array (1, 4) containing the plane parameters a,b,c,d
    """
    ctr = xyz.mean(axis=1)
    normalized = xyz - ctr[:, np.newaxis]
    M = np.dot(normalized, normalized.T)

    p = np.linalg.svd(M)[0][:, -1]
    d = np.matmul(p, ctr)

    p = np.append(p, -d)

    return p


def distance_to_plane(plane, x, y, z):
    
    """
    gets distance between points provided by their x, and y, z coordinates
    and a plane of the form ax+by+cz+d = 0

    Arguments:
        
    plane -- array (4,1) of plane parameter components [a,b,c,d]
    x -- array (N,1) x coordinates of the points
    y -- array (N,1) y coordinates of the points
    z -- array (N,1) z coordinates of the points

    Returns:
        
    distances -- array (N, 1) containing the distances between points and the plane
    """
    a, b, c, d = plane

    return (a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2)

