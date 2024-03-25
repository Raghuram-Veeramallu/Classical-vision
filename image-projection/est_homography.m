function [ H ] = est_homography(video_pts, logo_pts)
% est_homography estimates the homography to transform each of the
% video_pts into the logo_pts
% Inputs:
%     video_pts: a 4x2 matrix of corner points in the video
%     logo_pts: a 4x2 matrix of logo points that correspond to video_pts
% Outputs:
%     H: a 3x3 homography matrix such that logo_pts ~ H*video_pts
% Written for the University of Pennsylvania's Robotics:Perception course

video_pts = horzcat(video_pts, ones(size(video_pts, 1), 1));
ax = horzcat((video_pts * -1), zeros(size(video_pts, 1), 3), (video_pts .* logo_pts(:, 1)));
ay = horzcat(zeros(size(video_pts, 1), 3), (video_pts * -1), (video_pts .* logo_pts(:, 2)));

A = zeros(2 * size(video_pts, 1), 9);

A(1:2:end, :) = ax;
A(2:2:end, :) = ay;

[~, ~, V] = svd(A);

flatH = V(:, end);

H = reshape(flatH, [3,3])';  

end

