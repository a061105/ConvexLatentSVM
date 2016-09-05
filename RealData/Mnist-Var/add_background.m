function [img2, pos] = add_background(img, W_bg, H_bg)

img2 = rand(W_bg, H_bg);
[W,H] = size(img);

x = ceil(rand*(W_bg-W));
y = ceil(rand*(H_bg-H));

img2(x+1:x+W , y+1:y+H) = img;
img2 = img2-0.5;
pos = [x,y];
