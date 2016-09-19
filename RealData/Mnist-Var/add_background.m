function [img2, pos] = add_background(img, W_bg, H_bg, noise_level)

[W,H] = size(img);
img2 = zeros(W_bg, H_bg);

%add noise type 1
%img2 = rand(W_bg, H_bg)*noise_level;

%place image on the background
x = ceil(rand*(W_bg-W));
y = ceil(rand*(H_bg-H));
img2(x+1:x+W , y+1:y+H) = img;

%add noise type 2
num_points = floor(noise_level*W_bg*H_bg);
rand_x = ceil(rand(num_points,1)*W_bg);
rand_y = ceil(rand(num_points,1)*H_bg);
ind = (rand_x-1)*H_bg+rand_y;
img2(ind) = rand(num_points,1);

%img2 = img2-0.5;
pos = [x,y];
