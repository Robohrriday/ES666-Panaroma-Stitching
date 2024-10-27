import glob
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class PanaromaStitcher():
    def __init__(self):
        pass
    
    def detect_sift_features(self, img):

        """Reference: CV2 Sift Features: https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html"""
 
        gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()    
        kp, des = sift.detectAndCompute(gray,None)
        return (kp, des)
    
    def plot_image(self, img, gray=False):
        plt.figure()
        if gray:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    def make_panaroma_for_images_in(self, path):

        all_images = sorted(glob.glob(path+os.sep+'*'))
        print('Found {} Images for stitching'.format(len(all_images)))
        
        # Feature detection
        images_with_sift = []
        for i in range(len(all_images)):
            img = cv2.imread(all_images[i])
            images_with_sift.append(self.detect_sift_features(img))
        
        # Feature Matching
        bf = cv2.BFMatcher()
        matches = []
        for i in range(1, len(images_with_sift)):
            matches.append(bf.knnMatch(images_with_sift[0][1], images_with_sift[i][1], k=2)) # First image is reference
        
        # Ratio Test
        good_matches = []
        for match in matches:
            good = []
            for m,n in match:
                if m.distance < 0.75*n.distance:
                    good.append(m)
            good_matches.append(good)
        
        print(len(good_matches), type(good_matches[0][0]))
        


if __name__ == '__main__':
    ps = PanaromaStitcher()
    ps.make_panaroma_for_images_in('../../Images/I1')
        