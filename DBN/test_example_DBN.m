afunction test_example_DBN
load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);


rand('state',0)
dbn.sizes = [256 256 256 256 256];
opts.numepochs =   300;
opts.batchsize = 100;
opts.momentum  =   0.7;
opts.alpha     =   0.5;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

nn = dbnunfoldtonn(dbn, 10);
nn.activation_function = 'sigm';

opts.numepochs =  300;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);

