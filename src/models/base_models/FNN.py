from torch.nn import (
    Module, 
    Linear, 
    ReLU, 
    Sigmoid, 
    Softmax, 
    LeakyReLU,
    GELU, 
    Tanh,
    Dropout, 
    Sequential, 
    LayerNorm,
    BatchNorm1d
)
import logging

# Setting logger
extra = {'app_name':__name__}
logger = logging.getLogger(__name__)
syslog = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(app_name)s - %(levelname)s : %(message)s')
syslog.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(syslog)
logger = logging.LoggerAdapter(logger, extra)

ACTIVATION_FUNCTIONS = {
    'relu': ReLU(),
    'sigmoid': Sigmoid(),
    'tanh': Tanh(),
    'softmax': Softmax(dim=1),
    'leaky_relu': LeakyReLU(0.01),
    'gelu': GELU(),
}

LAYER_NORM = {
    'LayerNorm': LayerNorm,
    'BatchNorm1d': BatchNorm1d, 
}

CONFIG_MODEL_KEYS = ['activation_hidden', 'activation_output', 'p_dropout', 'norm']

DEFAULT_CONFIG = {
        'activation_hidden': None,
        'activation_output': None,
        'p_dropout': 0.2,
        'norm': None
}

def get_layer_name(layer, idx):
    layer_type = layer.__class__.__name__
    return f"{layer_type}_{idx}"

class FNN(Module):
    def __init__(self, input_size, hidden_layers, output_size, config_model=None):
        '''
        input_size: int, the size of inputs
        hidden_layers: list, a list with the size of each hidden layer
        output_size: int, the size of the output size
        config_model: dict, with the structure of
            {
                activation_hidden: <name_of_activation_function>,
                activation_output: <name_of_activation_output>,
                p_dropout: <probability for dropout>,
                norm: <name_of_layer_norm>
            }
        '''
        super(FNN, self).__init__()

        if config_model is None:
            config_model = {}
        
        not_keys = [x
                    for x in CONFIG_MODEL_KEYS
                    if x not in config_model.keys()]
        
        if len(not_keys)>0:
            logger.warning(f'{not_keys} keys not found in config_model. Setting them to default values')
            for config in not_keys:
                config_model[config] = DEFAULT_CONFIG[config]

        layers = []
        
        # Add input layers. If given, add hidden layers
        if hidden_layers:
            shapes = [input_size]+hidden_layers
            for i, o in zip(shapes[:-1], shapes[1:]):
                layers.append(Linear(i, o))
                
                if config_model['norm'] is not None:
                    layers.append(LAYER_NORM[config_model['norm']](o))
                if config_model['activation_hidden'] is not None:
                    layers.append(ACTIVATION_FUNCTIONS[config_model['activation_hidden']])
                if config_model['p_dropout'] is not None:
                    layers.append(Dropout(config_model['p_dropout']))
        else:
            o = input_size

        # Add output layer
        out = Linear(o, output_size)
        layers.append(out)
        if config_model['activation_output'] is not None:
            layers.append(ACTIVATION_FUNCTIONS[config_model['activation_output']])
        
        self.network = Sequential()

        for i, layer in enumerate(layers):
            layer_name = get_layer_name(layer, i)
            self.network.add_module(layer_name, layer)

    def forward(self, x):
        return self.network(x)