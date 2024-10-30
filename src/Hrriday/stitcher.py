import glob
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

class PanaromaStitcher():
    def __init__(self):
        self.final_stitch = None
        self.H_matrices = None
    
    def detect_sift_features(self, img):
        """
        Reference: CV2 Sift Features: https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
        """

        gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()    
        kp, des = sift.detectAndCompute(gray,None)
        return (kp, des)
    
    def plot_image(self, img, figsize = None, gray=False):
        if figsize != None:
            plt.figure(figsize=figsize)
        else:
            plt.figure()
        if gray:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    def cylindrical_warp(self, image, focal_length):
        """
        References: 
        https://courses.cs.washington.edu/courses/cse576/08sp/lectures/Stitching.pdf
        https://youtu.be/taty6lPVcmA?si=DVuhGKr-9DYet8gi&t=3557
        """
        h, w = image.shape[:2]
        x_c, y_c = w // 2, h // 2

        u, v = np.meshgrid(np.arange(w), np.arange(h))

        theta = (u - x_c) / focal_length
        h_cyl = (v - y_c) / focal_length

        x_hat = np.sin(theta)
        y_hat = h_cyl
        z_hat = np.cos(theta)

        x_img = (focal_length * x_hat / z_hat + x_c).astype(np.int32)
        y_img = (focal_length * y_hat / z_hat + y_c).astype(np.int32)

        valid_mask = (x_img >= 0) & (x_img < w) & (y_img >= 0) & (y_img < h)

        cylindrical_img = np.zeros_like(image)
        cylindrical_img[v[valid_mask], u[valid_mask]] = image[y_img[valid_mask], x_img[valid_mask]]

        cylindrical_img = Image.fromarray(cylindrical_img)
        cylindrical_img = cylindrical_img.crop((u[valid_mask].min(), v[valid_mask].min(), u[valid_mask].max(), v[valid_mask].max()))
        cylindrical_img = np.array(cylindrical_img)

        return cylindrical_img

    def RANSAC(self, query_coords, train_coords, epsilon = 5.0, s = 4, N = 1000):
        """
        RANSAC Algorithm to find the best homography matrix
        """
        best_inliers = []
        best_transform = None
        for i in tqdm(range(N)):
            # Randomly select s matches
            random_indices = np.random.choice(len(query_coords), s, replace=False)
            q = np.array([query_coords[i] for i in random_indices])
            t = np.array([train_coords[i] for i in random_indices])

            # Estimate homography matrix
            A = np.zeros((2*s, 9))
            for i in range(s):
                A[2*i] = np.array([-q[i][0], -q[i][1], -1, 0, 0, 0, q[i][0]*t[i][0], q[i][1]*t[i][0], t[i][0]])
                A[2*i+1] = np.array([0, 0, 0, -q[i][0], -q[i][1], -1, q[i][0]*t[i][1], q[i][1]*t[i][1], t[i][1]])
            
            U, S, V = np.linalg.svd(A)
            H = V[-1].reshape(3,3)
            
            q = np.hstack([np.array(query_coords), np.ones((len(query_coords), 1))])
            t_ = np.dot(H, q.T).T
            t_ = t_[:, :2] / t_[:, 2:]

            # Compute the number of inliers
            distances = np.linalg.norm(t_ - np.array(train_coords), axis=1)
            inliers = distances < epsilon
            if np.sum(inliers) > len(best_inliers):
                best_inliers = np.linspace(0, len(inliers), len(inliers), endpoint=False, dtype = np.int64)[inliers]
                best_transform = H
        
        A = np.zeros((2*len(best_inliers), 9))
        for i, inlier  in enumerate(best_inliers):
            A[2*i] = np.array([-query_coords[inlier][0], -query_coords[inlier][1], -1, 0, 0, 0, query_coords[inlier][0]*train_coords[inlier][0], query_coords[inlier][1]*train_coords[inlier][0], train_coords[inlier][0]])
            A[2*i+1] = np.array([0, 0, 0, -query_coords[inlier][0], -query_coords[inlier][1], -1, query_coords[inlier][0]*train_coords[inlier][1], query_coords[inlier][1]*train_coords[inlier][1], train_coords[inlier][1]])

        U, S, V = np.linalg.svd(A)
        best_transform = V[-1].reshape(3,3)
                                    
        return best_transform

    def single_weights_array(self, size: int) -> np.ndarray:
        """
        Reference: https://github.com/CorentinBrtx/image-stitching/tree/main
        """
        if size % 2 == 1:
            return np.concatenate(
                [np.linspace(0, 1, (size + 1) // 2), np.linspace(1, 0, (size + 1) // 2)[1:]]
            )
        else:
            return np.concatenate([np.linspace(0, 1, size // 2), np.linspace(1, 0, size // 2)])

    def single_weights_matrix(self, shape: tuple[int]) -> np.ndarray:
        """
        Reference: https://github.com/CorentinBrtx/image-stitching/tree/main
        """
        return (
            self.single_weights_array(shape[0])[:, np.newaxis]
            @ self.single_weights_array(shape[1])[:, np.newaxis].T
        )
    
    def compute_homography(self, path, a, b, cylinder_warp, focal_length, verbose = False):
        all_images = sorted(glob.glob(path+os.sep+'*'))
        if verbose:
            print('Found {} Images for stitching'.format(len(all_images)))

        query_image_idx = a
        train_image_idx = b

        # Feature detection
        images_with_sift = []
        for i in range(len(all_images)):
            img = cv2.imread(all_images[i])
            if cylinder_warp:
                img = self.cylindrical_warp(img, focal_length)
            images_with_sift.append(self.detect_sift_features(img))
        if verbose:
            print('Feature Detection Done')

        # Feature Matching
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(images_with_sift[query_image_idx][1], images_with_sift[train_image_idx][1], k=2)
        # Ratio Test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)
        if verbose:
            print('Feature Matching Done')

        # Feature Coodinates Extraction 
        query_coords = []
        train_coords = []
        for i in range(len(good)):
            query_coords.append(images_with_sift[query_image_idx][0][good[i].queryIdx].pt)
            train_coords.append(images_with_sift[train_image_idx][0][good[i].trainIdx].pt)
        if verbose:
            print('Feature Coordinates Extraction Done')

        # Homography Estimation
        H = self.RANSAC(query_coords, train_coords, epsilon = 5.0, s = 4, N = 1000)
        if verbose:
            print('Homography Estimation Done')
        return  H

    def stitch_images(self, H_matrices, images, reference_image_idx):
        
        # Left Stitching
        left_stitched_img = (images[0].copy()).astype(np.uint8)
        left_translation_matrices = []
        left_weight_matrix = self.single_weights_matrix(images[0].shape[:2])
        for i in range(0, reference_image_idx):
            H = H_matrices[i]
            H = H/H[-1,-1]
            train_corners = np.array([[0,0,1], [0,left_stitched_img.shape[0],1], [left_stitched_img.shape[1], left_stitched_img.shape[0],1], [left_stitched_img.shape[1], 0, 1]])
            if i != 0:
                warped_corners = np.dot(H @ np.linalg.inv(left_translation_matrices[-1]), train_corners.T).T
            else:
                warped_corners = np.dot(H, train_corners.T).T
            warped_corners = warped_corners/warped_corners[:,2].reshape(-1,1)
            query_corners = np.array([[0,0,1], [0,images[i+1].shape[0],1], [images[i+1].shape[1], images[i+1].shape[0],1], [images[i+1].shape[1], 0, 1]])
            all_corners = np.concatenate([query_corners[:, :2], warped_corners[:, :2]], axis=0)
            min_x = int(np.min(all_corners[:,0]))
            max_x = int(np.max(all_corners[:,0]))
            min_y = int(np.min(all_corners[:,1]))
            max_y = int(np.max(all_corners[:,1]))

            translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
            translation = translation.astype(np.float32)
            left_translation_matrices.append(translation.copy())
            if i != 0:
                train_warped = cv2.warpPerspective(left_stitched_img, left_translation_matrices[-1] @ H @ np.linalg.inv(left_translation_matrices[-2]), (max_x-min_x, max_y-min_y))
            else:
                train_warped = cv2.warpPerspective(left_stitched_img, left_translation_matrices[-1] @ H, (max_x-min_x, max_y-min_y))
            query_warped = cv2.warpPerspective(images[i+1], left_translation_matrices[-1], (max_x-min_x, max_y-min_y))

            mask1 = self.single_weights_matrix(images[i].shape[:2])
            mask2 = self.single_weights_matrix(images[i+1].shape[:2])
            if i != 0:
                train_mask = cv2.warpPerspective(left_weight_matrix, left_translation_matrices[-1] @ H @ np.linalg.inv(left_translation_matrices[-2]), (max_x-min_x, max_y-min_y))
            else:
                train_mask = cv2.warpPerspective(mask1, left_translation_matrices[-1] @ H, (max_x-min_x, max_y-min_y))
            query_mask = cv2.warpPerspective(mask2, left_translation_matrices[-1], (max_x-min_x, max_y-min_y))
            
            train_mask = (train_mask/train_mask.max()).astype(np.float32)
            query_mask = (query_mask/query_mask.max()).astype(np.float32)
            normalized_mask = np.divide(train_mask, train_mask+query_mask, where=(train_mask+query_mask)!=0)
            normalized_mask = normalized_mask/normalized_mask.max()
            normalized_mask = np.array([normalized_mask, normalized_mask, normalized_mask]).transpose(1,2,0)
            left_weight_matrix = (train_mask + query_mask)/(train_mask + query_mask).max()
            
            left_stitched_img = normalized_mask * train_warped + (1-normalized_mask) * query_warped
            left_stitched_img = left_stitched_img.astype(np.uint8)

            # del train_mask, query_mask, train_warped, query_warped, normalized_mask

        # Right Stitching
        right_stitched_img = (images[-1].copy()).astype(np.uint8)
        right_translation_matrices = []
        right_weight_matrix = self.single_weights_matrix(images[-1].shape[:2])
        for i in range(len(H_matrices)-1, reference_image_idx, -1):
            H = np.linalg.inv(H_matrices[i])
            H = H/H[-1,-1]
            train_corners = np.array([[0,0,1], [0,right_stitched_img.shape[0],1], [right_stitched_img.shape[1], right_stitched_img.shape[0],1], [right_stitched_img.shape[1], 0, 1]])
            if i != len(H_matrices)-1:
                warped_corners = np.dot(H @ np.linalg.inv(right_translation_matrices[-1]), train_corners.T).T
            else:
                warped_corners = np.dot(H, train_corners.T).T
            warped_corners = warped_corners/warped_corners[:,2].reshape(-1,1)
            query_corners = np.array([[0,0,1], [0,images[i].shape[0],1], [images[i].shape[1], images[i].shape[0],1], [images[i].shape[1], 0, 1]])
            all_corners = np.concatenate([query_corners[:, :2], warped_corners[:, :2]], axis=0)
            min_x = int(np.min(all_corners[:,0]))
            max_x = int(np.max(all_corners[:,0]))
            min_y = int(np.min(all_corners[:,1]))
            max_y = int(np.max(all_corners[:,1]))

            translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
            translation = translation.astype(np.float32)
            right_translation_matrices.append(translation.copy())
            if i != len(H_matrices)-1:
                train_warped = cv2.warpPerspective(right_stitched_img, right_translation_matrices[-1] @ H @ np.linalg.inv(right_translation_matrices[-2]), (max_x-min_x, max_y-min_y))
            else:
                train_warped = cv2.warpPerspective(right_stitched_img, right_translation_matrices[-1] @ H, (max_x-min_x, max_y-min_y))
            query_warped = cv2.warpPerspective(images[i], right_translation_matrices[-1], (max_x-min_x, max_y-min_y))
            
            mask1 = self.single_weights_matrix(images[i+1].shape[:2])
            mask2 = self.single_weights_matrix(images[i].shape[:2])
            if i != len(H_matrices)-1:
                train_mask = cv2.warpPerspective(right_weight_matrix, right_translation_matrices[-1] @ H @ np.linalg.inv(right_translation_matrices[-2]), (max_x-min_x, max_y-min_y))
            else:
                train_mask = cv2.warpPerspective(mask1, right_translation_matrices[-1] @ H, (max_x-min_x, max_y-min_y))
            query_mask = cv2.warpPerspective(mask2, right_translation_matrices[-1], (max_x-min_x, max_y-min_y))
            
            train_mask = (train_mask/train_mask.max()).astype(np.float32)
            query_mask = (query_mask/query_mask.max()).astype(np.float32)
            normalized_mask = np.divide(train_mask, train_mask+query_mask, where=(train_mask+query_mask)!=0)
            normalized_mask = normalized_mask/normalized_mask.max()
            normalized_mask = np.array([normalized_mask, normalized_mask, normalized_mask]).transpose(1,2,0)
            right_weight_matrix = (train_mask + query_mask)/(train_mask + query_mask).max()

            right_stitched_img = normalized_mask * train_warped + (1-normalized_mask) * query_warped
            right_stitched_img = right_stitched_img.astype(np.uint8)

            # del train_mask, query_mask, train_warped, query_warped, normalized_mask
        
        # Final Stitching
        if reference_image_idx == len(H_matrices):
            return left_stitched_img
        else:
            H = np.linalg.inv(H_matrices[reference_image_idx])
            H = H/H[-1,-1]
            train_corners = np.array([[0,0,1], [0,right_stitched_img.shape[0],1], [right_stitched_img.shape[1], right_stitched_img.shape[0],1], [right_stitched_img.shape[1], 0, 1]])
            if reference_image_idx != len(H_matrices)-1:
                warped_corners = np.dot(H @ np.linalg.inv(right_translation_matrices[-1]), train_corners.T).T
            else:
                warped_corners = np.dot(H, train_corners.T).T
            warped_corners = warped_corners/warped_corners[:,2].reshape(-1,1)
            query_corners = np.array([[0,0,1], [0,left_stitched_img.shape[0],1], [left_stitched_img.shape[1], left_stitched_img.shape[0],1], [left_stitched_img.shape[1], 0, 1]])
            if reference_image_idx != 0:
                query_corners = np.dot(np.linalg.inv(left_translation_matrices[-1]), query_corners.T).T
            all_corners = np.concatenate([query_corners[:, :2], warped_corners[:, :2]], axis=0)
            min_x = int(np.min(all_corners[:,0]))
            max_x = int(np.max(all_corners[:,0]))
            min_y = int(np.min(all_corners[:,1]))
            max_y = int(np.max(all_corners[:,1]))

            translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
            translation = translation.astype(np.float32)
            right_translation_matrices.append(translation.copy())
            if reference_image_idx != len(H_matrices)-1:
                train_warped = cv2.warpPerspective(right_stitched_img, right_translation_matrices[-1] @ H @ np.linalg.inv(right_translation_matrices[-2]), (max_x-min_x, max_y-min_y))
            else:
                train_warped = cv2.warpPerspective(right_stitched_img, right_translation_matrices[-1] @ H, (max_x-min_x, max_y-min_y))
            if reference_image_idx != 0:
                query_warped = cv2.warpPerspective(left_stitched_img, right_translation_matrices[-1] @ np.linalg.inv(left_translation_matrices[-1]), (max_x-min_x, max_y-min_y))
            else:
                query_warped = cv2.warpPerspective(left_stitched_img, right_translation_matrices[-1], (max_x-min_x, max_y-min_y))
            
            mask = self.single_weights_matrix(images[-1].shape[:2])
            if reference_image_idx != len(H_matrices)-1:
                train_mask = cv2.warpPerspective(right_weight_matrix, right_translation_matrices[-1] @ H @ np.linalg.inv(right_translation_matrices[-2]), (max_x-min_x, max_y-min_y))
            else:
                train_mask = cv2.warpPerspective(mask, right_translation_matrices[-1] @ H, (max_x-min_x, max_y-min_y))
            if reference_image_idx != 0:
                query_mask = cv2.warpPerspective(left_weight_matrix, right_translation_matrices[-1] @ np.linalg.inv(left_translation_matrices[-1]), (max_x-min_x, max_y-min_y))
            else:
                query_mask = cv2.warpPerspective(mask, right_translation_matrices[-1], (max_x-min_x, max_y-min_y))
            
            train_mask = (train_mask/train_mask.max()).astype(np.float32)
            query_mask = (query_mask/query_mask.max()).astype(np.float32)
            normalized_mask = np.divide(train_mask, train_mask+query_mask, where=(train_mask+query_mask)!=0)
            normalized_mask = normalized_mask/normalized_mask.max()
            normalized_mask = np.array([normalized_mask, normalized_mask, normalized_mask]).transpose(1,2,0)
            # final_weight_matrix = (train_mask + query_mask)/(train_mask + query_mask).max()

            final_stitched_img = normalized_mask * train_warped + (1-normalized_mask) * query_warped
            final_stitched_img = final_stitched_img.astype(np.uint8)

            del train_mask, query_mask, train_warped, query_warped, normalized_mask

        return final_stitched_img    
    
    def make_panaroma_for_images_in(self, path, reference_image_idx = 2, cylinder_warp = True, focal_length_ratio = 2):
        H_matrices = []
        for i in range(len(os.listdir(path))-1):
            H = self.compute_homography(path, i, i+1, cylinder_warp, focal_length_ratio, verbose = False)
            H_matrices.append(H)
        
        all_images = sorted(glob.glob(path+os.sep+'*'))
        images = []
        for i in range(len(all_images)):
            img = cv2.imread(all_images[i])
            if cylinder_warp:
                img = self.cylindrical_warp(img, focal_length_ratio)
            images.append(img)
        
        self.final_stitch = self.stitch_images(H_matrices, images, reference_image_idx)
        self.H_matrices = H_matrices
        return self.final_stitch, self.H_matrices

if __name__ == '__main__':
    ps = PanaromaStitcher()
    final_stitch, H_matrices = ps.make_panaroma_for_images_in('../../Images/I6', reference_image_idx=2, cylinder_warp = True, focal_length_ratio=2)
    # ps.plot_image(ps.final_stitch, figsize=(20,10))