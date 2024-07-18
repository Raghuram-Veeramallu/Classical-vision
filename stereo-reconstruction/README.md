## Stereo Reconstruction

This project performs stereo reconstruction using a pair of stereo images, computes the SIFT descriptors, matches the SIFT features and computes the fundamental matrix (F) using 8-point RANSAC. Using the F matrix, the camera poses are computed. The reconstructed 3D points are calculated using triangulation. The best camera pose out of the possible poses using triangulation and then stereo matching is performed between the two views.


<img src="results/orig_left_right.png" width=100% height=100% alt="original">
<center>Original Left and Right images</center>  
<br>
<img src="results/sift_feature_matching.png" width=49.5% height=49% alt="original">   <img src="results/epipolar_lines.png" width=49% height=% alt="original">
<center>SIFT Feature Matching and Epipolar lines</center>  
<br>
<img src="results/camera_configurations.png" width=49.5% height=49% alt="original">   <img src="results/camera_poses_w_pcl.png" width=49.5% height=49% alt="original">
<center>Camera Configuations computed from F along with PCL visualizations</center>  
<br>

<figure>
    <img src="results/stereo_rectification.png" width=100% height=100% alt="zoom">
    <center><figcaption>Stereo Rectification</figcaption></center>
</figure>

<figure>
    <img src="results/disparity_map.png" width=100% height=100% alt="zoom">
    <center><figcaption>Disparity Map of the stereo reconstruction</figcaption></center>
</figure>

Done as part of the [CSCI 5561: Computer Vision](https://www-users.cse.umn.edu/~hspark/csci5561_F2020/csci5561.html) course requirements.
