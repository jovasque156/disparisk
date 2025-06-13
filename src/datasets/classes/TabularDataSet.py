import torch
from torch.utils.data import Dataset
from torch import no_grad as torch_no_grad
from torch import tensor as torch_tensor
from torch import from_numpy as torch_from_numpy
from torch import long as torch_long

class TabularDataSet(Dataset):
    def __init__(self, tabular_df, features, encoder=None, encoder_config = None):
        '''
        tabular_df: pandas, dataframe with the input_space, sensitive, and label features
        features: dict, a dictionary with the structure
                    {
                        'input_space': <name_cols>,
                        'sensitive': <name_col>,
                        'label': <name_col>
                    }
        encoder: Model.nn class, encoder that encode the input_space
        encoder_config: dict, a dictionary with the structure
                    {
                        'input_cols': <name_cols>,
                        'id_output_to_use': <id_output_to_use> #leave as None if there's just one output
                    }
        '''

        self.tabular_df = tabular_df
        self.features = features
        self.encoder = encoder
        self.encoder_config = encoder_config
        self.indexes = tabular_df.index

    def __len__(self):
        return len(self.tabular_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # obtain input depending if encoder is given or not
        if self.encoder is None:
            input_ = self.tabular_df.loc[self.indexes[idx],
                                            self.features['input_space']].astype('float32')
            input_ = torch_from_numpy(input_.values)
        else:
            input_ = self.tabular_df.loc[self.indexes[idx],
                                            self.encoder_config['input_cols']]
            with torch_no_grad:
                input_ = self.encoder(input_) if self.encoder_config['id_output_to_use'] is None \
                            else self.encoder(input_)[self.encoder_config['id_output_to_use']]

        label = torch_tensor(self.tabular_df.loc[self.indexes[idx], self.features['label']],
                                dtype=torch_long)
        sensitive = torch_tensor(self.tabular_df.loc[self.indexes[idx], self.features['sensitive']],
                                dtype=torch_long)    

        return {'X': input_, 'label': label, 'sensitive': sensitive}
        
        



