config_FNN = {
  'h1': {
        'n_hidden_layers': 1,
        'activation_hidden': None,
        'activation_output': None,
        'p_dropout': 0.2,
        'norm': 'BatchNorm1d'
      },
  'h4': {
        'n_hidden_layers': 4,
        'activation_hidden': None,
        'activation_output': None,
        'p_dropout': 0.2,
        'norm': 'BatchNorm1d'
      },
  'h1_relu': {
        'n_hidden_layers': 1,
        'activation_hidden': 'relu',
        'activation_output': None,
        'p_dropout': 0.2,
        'norm': 'BatchNorm1d'
      },
  'h4_relu': {
        'n_hidden_layers': 4,
        'activation_hidden': 'relu',
        'activation_output': None,
        'p_dropout': 0.2,
        'norm': 'BatchNorm1d'
      },
  'h1_sigmoid': {
        'n_hidden_layers': 1,
        'activation_hidden': 'sigmoid',
        'activation_output': None,
        'p_dropout': 0.2,
        'norm': 'BatchNorm1d'
      },
  'h4_sigmoid': {
        'n_hidden_layers': 4,
        'activation_hidden': 'sigmoid',
        'activation_output': None,
        'p_dropout': 0.2,
        'norm': 'BatchNorm1d'
      },
  'h1_gelu': {
        'n_hidden_layers': 1,
        'activation_hidden': 'gelu',
        'activation_output': None,
        'p_dropout': 0.2,
        'norm': 'BatchNorm1d'
      },
  'h4_gelu': {
      'n_hidden_layers': 4,
      'activation_hidden': 'gelu',
      'activation_output': None,
      'p_dropout': 0.2,
      'norm': 'BatchNorm1d'
    }
}