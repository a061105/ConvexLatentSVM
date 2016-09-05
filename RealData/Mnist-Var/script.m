pos_label = 2;
neg_label = 3;
H = 28;
W = 28;
dim = W*H;
truncate_thd = 1e-2;
rotate_interval = 30;
num_rotate = ceil(360/rotate_interval)-1;
%output names
svm_train_fname = ['mnist_rot.' num2str(pos_label) 'vs' num2str(neg_label) '.train'];
svm_test_fname = ['mnist_rot.' num2str(pos_label) 'vs' num2str(neg_label) '.test'];
latentsvm_train_fname = [svm_train_fname '.latent'];
latentsvm_test_fname = [svm_test_fname '.latent'];

A = load('mnist_all_rotation_normalized_float_train_valid.amat');
A_train = A(1:2000,:);
A_test = A(2001:end,:);
%A_test = load('mnist_all_rotation_normalized_float_test.amat');

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

for k = 1:length(X_list)
	X = X_list{k};
	y = y_list{k};
	N = size(X,1);
	
	fname = fname_list{k}
	fp = fopen(fname,'w');
	fprintf(fp, '%d\n', N);
	for i = 1:N
		fprintf(fp, '%d, ', y(i));
		
		img = reshape(X(i,:), [W H]);
		for j = 0:num_rotate
			write_x_libsvm( fp, reshape(img,[1 dim]), truncate_thd);
			fprintf(fp, ' . ');
			img = imrotate(img, rotate_interval, 'bilinear', 'crop');
		end
		fprintf(fp,'\n');
	end
end
