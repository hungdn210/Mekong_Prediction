import torch


class BasicConfigs:
    def __init__(self):
        # basic config
        self._is_training = True
        self._model = 'Linear'
        # data loader
        self._data = 'MeKong'
        self._root_path = './dataset/Filled_Gaps_Data/Water.Level/'
        self._data1_path = 'Stung Treng.csv'
        self._data2_path = 'Kratie.csv'
        self._features = 'M'
        self._target = 'Value'
        self._freq = 'd'
        self._checkpoints = './checkpoints/'
        self._seq_len = 3
        self._label_len = 0
        self._pred_len = 1
        self._inverse = True
        # model define
        self._num_kernels = 6
        self._enc_in = 1
        self._dec_in = 1
        self._c_out = 1
        self._d_model = 512
        self._n_heads = 8
        self._e_layers = 1
        self._d_layers = 1
        self._d_ff = 2048
        self._moving_avg = 25
        self._factor = 1
        self._distil = True
        self._dropout = 0.1
        self._embed = 'timeF'
        self._activation = 'gelu'
        # optimization
        self._num_workers = 10
        self._itr = 1
        self._train_epochs = 100
        self._batch_size = 32
        self._patience = 3
        self._learning_rate = 0.0001
        self._des = ''
        self._loss = 'MSE'
        self._lradj = 'type1'
        self._use_amp = False
        self._use_gpu = True if torch.cuda.is_available() else False
        self._gpu = 0
        self._use_multi_gpu = False
        self._devices = '0,1,2,3'

    def model_default_configs(self, model):
        dic = {
            'Linear': {
                'learning_rate': 0.1,
                'd_model': 128,
                'd_ff': 512,
                'n_heads': 2,
                'e_layers': 2,
                'd_layers': 1,
            },
            'iTransformer': {
                'learning_rate': 0.001,
                'd_model': 128,
                'd_ff': 512,
                'n_heads': 2,
                'e_layers': 2,
                'd_layers': 1,
            },
            'PatchTST': {
                'learning_rate': 0.001,
                'd_model': 128,
                'd_ff': 512,
                'n_heads': 2,
                'e_layers': 2,
                'd_layers': 1,
            },
        }
        self._learning_rate = dic[model]['learning_rate']
        self._d_model = dic[model]['d_model']
        self._d_ff = dic[model]['d_ff']
        self._n_heads = dic[model]['n_heads']
        self._e_layers = dic[model]['e_layers']
        self._d_layers = dic[model]['d_layers']

    # Getters
    @property
    def is_training(self):
        """status"""
        return self._is_training

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        """dataset type: [MeKong, MeKong_Cross]"""
        return self._data

    @property
    def root_path(self):
        """root path of the data file"""
        return self._root_path

    @property
    def data1_path(self):
        """data1 file"""
        return self._data1_path

    @property
    def data2_path(self):
        """data2 file, only works when data==MeKong_Cross"""
        return self._data2_path

    @property
    def features(self):
        """forecasting task, options:[M, S, MS];
        M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate"""
        return self._features

    @property
    def target(self):
        """target feature in S or MS task"""
        return self._target

    @property
    def freq(self):
        """freq for time features encoding,
        options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly],
        you can also use more detailed freq like 15min or 3h"""
        return self._freq

    @property
    def checkpoints(self):
        """location of model checkpoints"""
        return self._checkpoints

    @property
    def seq_len(self):
        """input sequence length"""
        return self._seq_len

    @property
    def label_len(self):
        """start token length"""
        return self._label_len

    @property
    def pred_len(self):
        """prediction sequence length"""
        return self._pred_len

    @property
    def inverse(self):
        """inverse output data"""
        return self._inverse

    @property
    def num_kernels(self):
        """for Inception"""
        return self._num_kernels

    @property
    def enc_in(self):
        """encoder input size"""
        return self._enc_in

    @property
    def dec_in(self):
        """decoder input size"""
        return self._dec_in

    @property
    def c_out(self):
        """output size"""
        return self._c_out

    @property
    def d_model(self):
        """dimension of model"""
        return self._d_model

    @property
    def n_heads(self):
        """num of heads"""
        return self._n_heads

    @property
    def e_layers(self):
        """num of encoder layers"""
        return self._e_layers

    @property
    def d_layers(self):
        """num of decoder layers"""
        return self._d_layers

    @property
    def d_ff(self):
        """dimension of fcn"""
        return self._d_ff

    @property
    def moving_avg(self):
        """window size of moving average"""
        return self._moving_avg

    @property
    def factor(self):
        """attn factor"""
        return self._factor

    @property
    def distil(self):
        """whether to use distilling in encoder, using this argument means not using distilling"""
        return self._distil

    @property
    def dropout(self):
        """dropout"""
        return self._dropout

    @property
    def embed(self):
        """time features encoding, options:[timeF, fixed, learned]"""
        return self._embed

    @property
    def activation(self):
        """activation"""
        return self._activation

    @property
    def num_workers(self):
        """data loader num workers"""
        return self._num_workers

    @property
    def itr(self):
        """experiments times"""
        return self._itr

    @property
    def train_epochs(self):
        """train epochs"""
        return self._train_epochs

    @property
    def batch_size(self):
        """batch size of train input data"""
        return self._batch_size

    @property
    def patience(self):
        """early stopping patience"""
        return self._patience

    @property
    def learning_rate(self):
        """optimizer learning rate"""
        return self._learning_rate

    @property
    def des(self):
        """exp description"""
        return self._des

    @property
    def loss(self):
        """loss function"""
        return self._loss

    @property
    def lradj(self):
        """adjust learning rate"""
        return self._lradj

    @property
    def use_amp(self):
        """use automatic mixed precision training"""
        return self._use_amp

    @property
    def use_gpu(self):
        """use gpu"""
        return self._use_gpu

    @property
    def gpu(self):
        """gpu"""
        return self._gpu

    @property
    def use_multi_gpu(self):
        """use multiple gpus"""
        return self._use_multi_gpu

    @property
    def devices(self):
        """device ids of multiple gpus"""
        return self._devices

    # Setters
    @is_training.setter
    def is_training(self, is_training):
        assert isinstance(is_training, bool), "is_training must be bool"
        self._is_training = is_training

    @model.setter
    def model(self, model):
        assert isinstance(model, str), "model must be str"
        self._model = model

    @data.setter
    def data(self, data):
        assert isinstance(data, str), "data must be str"
        assert data in ['MeKong', 'MeKong_Cross'], "data must be MeKong or MeKong_Cross"
        self._data = data

    @root_path.setter
    def root_path(self, root_path):
        assert isinstance(root_path, str), "root_path must be str"
        self._root_path = root_path
    
    @data1_path.setter
    def data1_path(self, data1_path):
        assert isinstance(data1_path, str), "data1_path must be str"
        self._data1_path = data1_path
   
    @data2_path.setter
    def data2_path(self, data2_path):
        assert isinstance(data2_path, str), "data2_path must be str"
        self._data2_path = data2_path
   
    @features.setter
    def features(self, features):
        assert isinstance(features, str), "features must be str"
        assert features in ['M', 'S', 'MS'], "features must be M or S or MS"
        self._features = features
    
    @target.setter
    def target(self, target):
        assert isinstance(target, str), "target must be str"
        self._target = target
    
    @freq.setter
    def freq(self, freq):
        assert isinstance(freq, str), "freq must be str"
        assert freq in ['s', 't', 'h', 'd', 'b', 'w', 'm'], "freq must be either 's', 't', 'h', 'd', 'b', 'w' or 'm'"
        self._freq = freq
    
    @checkpoints.setter
    def checkpoints(self, checkpoints):
        assert isinstance(checkpoints, str), "checkpoints must be str"
        self._checkpoints = checkpoints
    
    @seq_len.setter
    def seq_len(self, seq_len):
        assert isinstance(seq_len, int), "seq_len must be int"
        assert seq_len > 0, "seq_len must be greater than 0"
        self._seq_len = seq_len
    
    @label_len.setter
    def label_len(self, label_len):
        assert isinstance(label_len, int), "label_len must be int"
        assert label_len > 0, "label_len must be greater than 0"
        self._label_len = label_len
    
    @pred_len.setter
    def pred_len(self, pred_len):
        assert isinstance(pred_len, int), "pred_len must be int"
        assert pred_len > 0, "pred_len must be greater than 0"
        self._pred_len = pred_len
    
    @inverse.setter
    def inverse(self, inverse):
        assert isinstance(inverse, bool), "inverse must be bool"
        self._inverse = inverse
    
    @num_kernels.setter
    def num_kernels(self, num_kernels):
        assert isinstance(num_kernels, int), "num_kernels must be int"
        assert num_kernels > 0, "num_kernels must be greater than 0"
        self._num_kernels = num_kernels
    
    @enc_in.setter
    def enc_in(self, enc_in):
        assert isinstance(enc_in, int), "enc_in must be int"
        assert enc_in > 0, "enc_in must be greater than 0"
        self._enc_in = enc_in
    
    @dec_in.setter
    def dec_in(self, dec_in):
        assert isinstance(dec_in, int), "dec_in must be int"
        assert dec_in > 0, "dec_in must be greater than 0"
        self._dec_in = dec_in
   
    @c_out.setter
    def c_out(self, c_out):
        assert isinstance(c_out, int), "c_out must be int"
        assert c_out > 0, "c_out must be greater than 0"
        self._c_out = c_out
    
    @d_model.setter
    def d_model(self, d_model):
        assert isinstance(d_model, int), "d_model must be int"
        assert d_model > 0, "d_model must be greater than 0"
        self._d_model = d_model
    
    @n_heads.setter
    def n_heads(self, n_heads):
        assert isinstance(n_heads, int), "n_heads must be int"
        assert n_heads > 0, "n_heads must be greater than 0"
        self._n_heads = n_heads
    
    @e_layers.setter
    def e_layers(self, e_layers):
        assert isinstance(e_layers, int), "e_layers must be int"
        assert e_layers > 0, "e_layers must be greater than 0"
        self._e_layers = e_layers
   
    @d_layers.setter
    def d_layers(self, d_layers):
        assert isinstance(d_layers, int), "d_layers must be int"
        assert d_layers > 0, "d_layers must be greater than 0"
        self._d_layers = d_layers
    
    @d_ff.setter
    def d_ff(self, d_ff):
        assert isinstance(d_ff, int), "d_ff must be int"
        assert d_ff > 0, "d_ff must be greater than 0"
        self._d_ff = d_ff
    
    @moving_avg.setter
    def moving_avg(self, moving_avg):
        assert isinstance(moving_avg, int), "moving_avg must be int"
        assert moving_avg > 0, "moving_avg must be greater than 0"
        self._moving_avg = moving_avg
    
    @factor.setter
    def factor(self, factor):
        assert isinstance(factor, int), "factor must be int"
        assert factor > 0, "factor must be greater than 0"
        self._factor = factor
   
    @distil.setter
    def distil(self, distil):
        assert isinstance(distil, bool), "distil must be bool"
        self._distil = distil
    
    @dropout.setter
    def dropout(self, dropout):
        assert isinstance(dropout, float) or isinstance(dropout, int), "dropout must be float or int"
        assert dropout >= 0, "dropout must be greater than 0"
        assert dropout <= 1, "dropout must be less than 1"
        self._dropout = dropout
    
    @embed.setter
    def embed(self, embed):
        assert isinstance(embed, str), "embed must be str"
        assert embed in ['timeF', 'fixed', 'learned'], "embed must be either 'timeF', 'fixed' or 'learned'"
        self._embed = embed 
   
    @activation.setter
    def activation(self, activation):
        assert isinstance(activation, str), "activation must be str"
        self._activation = activation
    
    @num_workers.setter
    def num_workers(self, num_workers):
        assert isinstance(num_workers, int), "num_workers must be int"
        assert num_workers > 0, "num_workers must be greater than 0"
        self._num_workers = num_workers

    @train_epochs.setter
    def train_epochs(self, train_epochs):
        assert isinstance(train_epochs, int), "train_epochs must be int"
        assert train_epochs > 0, "train_epochs must be greater than 0"
        self._train_epochs = train_epochs

    @batch_size.setter
    def batch_size(self, batch_size):
        assert isinstance(batch_size, int), "batch_size must be int"
        assert batch_size > 0, "batch_size must be greater than 0"
        self._batch_size = batch_size

    @patience.setter
    def patience(self, patience):
        assert isinstance(patience, int), "patience must be int"
        assert patience > 0, "patience must be greater than 0"
        self._patience = patience

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        assert isinstance(learning_rate, float), "learning_rate must be float"
        assert learning_rate > 0, "learning_rate must be greater than 0"
        self._learning_rate = learning_rate

    @des.setter
    def des(self, des):
        assert isinstance(des, int), "des must be int"
        self._des = des

    @loss.setter
    def loss(self, loss):
        assert isinstance(loss, str), "loss must be str"
        self._loss = loss

    @lradj.setter
    def lradj(self, lradj):
        assert isinstance(lradj, str), "lradj must be str"
        self._lradj = lradj

    @use_amp.setter
    def use_amp(self, use_amp):
        assert isinstance(use_amp, bool), "use_amp must be bool"
        self._use_amp = use_amp

    @use_gpu.setter
    def use_gpu(self, use_gpu):
        assert isinstance(use_gpu, bool), "use_gpu must be bool"
        self._use_gpu = use_gpu

    @gpu.setter
    def gpu(self, gpu):
        assert isinstance(gpu, int), "gpu must be int"
        self._gpu = gpu

    @use_multi_gpu.setter
    def use_multi_gpu(self, use_multi_gpu):
        assert isinstance(use_multi_gpu, bool), "use_multi_gpu must be bool"
        self._use_multi_gpu = use_multi_gpu

    @devices.setter
    def devices(self, devices):
        assert isinstance(devices, str), "devices must be str"
        self._devices = devices