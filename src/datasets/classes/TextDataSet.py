from torch.utils.data import Dataset
from torch import tensor as torch_tensor
from torch import long as torch_long
from torch import is_tensor as torch_is_tensor

class TextDataSet(Dataset):
    def __init__(self, text_df, features, tokenizer, config_tokenizer=None):
        '''
        df: pandas, dataframe with the texts and values to analyze
        features: dict, a dictionary with the structure
                {
                    'text': <name_col>,
                    'sensitive': <name_col>,
                    'label': <name_col>
                }
        tokenizer: tokenizer of the model to be used
        config_tokenizer: dict, a dictionary with the structure
                            {
                                'padding': <config>,
                                'max_length': <config>,
                                'truncation': <config>,
                            }
        '''
        self.text_df = text_df
        self.indexes = text_df.index
        self.text_df.index = range(len(text_df))
        self.features = features
        self.tokenizer = tokenizer
        self.config_tokenizer = config_tokenizer if config_tokenizer else {}

    def __len__(self):
        return len(self.text_df)

    def __getitem__(self, idx):
        if torch_is_tensor(idx):
            idx = idx.tolist()

        # Extract text, label, and sensitive features
        text = self.text_df.loc[idx, self.features['text']]
        label = self.text_df.loc[idx, self.features['label']]
        sensitive = self.text_df.loc[idx, self.features['sensitive']]

        if not isinstance(text, str):
            text = str(text) if text is not None else ''

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            padding=self.config_tokenizer.get('padding', 'max_length'),
            max_length=self.config_tokenizer.get('max_length', 512),
            truncation=self.config_tokenizer.get('truncation', True),
            return_tensors='pt'
        )

        # Return a dictionary with tokenized data and labels
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # Remove the batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch_tensor(self.features['label_dic'][label], dtype=torch_long),
            'sensitive': torch_tensor(self.features['sensitive_dic'][sensitive], dtype=torch_long)
        }