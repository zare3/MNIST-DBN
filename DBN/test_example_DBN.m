function test_example_DBN
load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

rand('state',0)
%train dbn
dbn.sizes = [1024 1024 1024];
opts.numepochs =   100;
opts.batchsize = 100;
opts.momentum  =   0.7;
opts.alpha     =   0.05;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 10);
nn.activation_function = 'sigm';

%train nn
opts.numepochs =  300;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);

disp ('TEST ERROR IS :')
disp (er)

assert(er < 0.10, 'Too big error');
