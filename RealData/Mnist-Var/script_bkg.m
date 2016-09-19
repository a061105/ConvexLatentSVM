pos_label = 0;
neg_label = 1;
H = 28;
W = 28;
dim = W*H;
truncate_thd = 1e-4;
noise_level = 1e-1;

W_bg = 100;
H_bg = 100;
trans_interval = 5;
num_window_height = floor((H_bg-H)/trans_interval);

%output names
svm_train_fname = ['mnist_bg.' num2str(pos_label) 'vs' num2str(neg_label) '.train'];
svm_test_fname = ['mnist_bg.' num2str(pos_label) 'vs' num2str(neg_label) '.test'];
latentsvm_train_fname = [svm_train_fname '.latent'];
latentsvm_test_fname = [svm_test_fname '.latent'];

%A = load('mnist_background_random_train.amat');
A = load('mnist_train.amat');
A_train = A(1:250,:);
A_test = A(250:750,:);

y_tr = A_train(:,end); %ignore validation for now
y_ts = A_test(:,end);

X_tr = A_train(:,1:end-1);
X_ts = A_test(:,1:end-1);
%truncate values close to 0
X_tr(abs(X_tr)<truncate_thd)=0;
X_ts(abs(X_ts)<truncate_thd)=0;

X_1vs1_tr = [X_tr(y_tr==pos_label,:); X_tr(y_tr==neg_label,:)];
X_1vs1_ts = [X_ts(y_ts==pos_label,:); X_ts(y_ts==neg_label,:)];

y_1vs1_tr = [ ones(nnz(y_tr==pos_label),1); -ones(nnz(y_tr==neg_label),1) ];
y_1vs1_ts = [ ones(nnz(y_ts==pos_label),1); -ones(nnz(y_ts==neg_label),1) ];

% write original data in libsvm format
libsvmwrite(svm_train_fname, y_1vs1_tr, sparse(X_1vs1_tr));
libsvmwrite(svm_test_fname, y_1vs1_ts, sparse(X_1vs1_ts));

% create data with different hidden rotations (to be discovered by LatentSVM model)
X_list = {X_1vs1_tr, X_1vs1_ts};
y_list = {y_1vs1_tr, y_1vs1_ts};
fname_list = {latentsvm_train_fname, latentsvm_test_fname};
fpos_list = {[latentsvm_train_fname '.pos'], [latentsvm_test_fname '.pos']};
%X_list = {X_1vs1_ts};
%y_list = {y_1vs1_ts};
%fname_list = {latentsvm_test_fname};

for k = 1:length(X_list)
	X = X_list{k};
	y = y_list{k};
	N = size(X,1);
	
	fname = fname_list{k}
	fpos_name = fpos_list{k};
	fp = fopen(fname,'w');
	fp2 = fopen([fname '-2'],'w');
	fp_pos = fopen(fpos_name, 'w');
	fprintf(fp, '%d\n', N);
	%fprintf(fp2, '%d\n', N);
	for i = 1:N
		fprintf(fp, '%d, ', y(i));
		fprintf(fp2, '%d ', y(i));
		
		img = reshape(X(i,:), [W H]);
		[img_bg, pos] = add_background(img, W_bg, H_bg, noise_level);
		
		min_dist = inf;
		argmin_count = -1;
		argmin_pos = -1;
		count = 0;
		for w = 1:trans_interval:W_bg-W+1
			for h=1:trans_interval:H_bg-H+1
				x = reshape(img_bg(w:w+W-1, h:h+H-1), [1,dim]);
				x = x / sqrt(754);
				write_x_libsvm( fp, x, truncate_thd);
				fprintf(fp, ' . ');
				
				dist_to_gt = norm([w-pos(1), h-pos(2)]);
				if dist_to_gt < min_dist 
					min_dist = dist_to_gt;
					argmin_count = count;
					argmin_pos = [w,h];
				end
				count = count + 1;
			end
		end
		
		%w = argmin_pos(1);
		%h = argmin_pos(2);
		%x = reshape(img_bg(w:w+W-1, h:h+H-1), [1,dim]);
		x = reshape(img_bg(:,:), [1,W_bg*H_bg]);
		x = x / sqrt(754);
		write_x_libsvm( fp2, x, truncate_thd );
		
		fprintf(fp_pos, '%d\n', argmin_count);
		
		%imshow(img_bg(argmin_pos(1):argmin_pos(1)+W-1, argmin_pos(2):argmin_pos(2)+H-1));
		%saveas(gcf, ['~/public_html/tmp/mnist_bg/' num2str(i) '.pdf'],'pdf');
		
		fprintf(fp,'\n');
		fprintf(fp2,'\n');
	end
end
