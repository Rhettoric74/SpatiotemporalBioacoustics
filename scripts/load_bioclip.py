import open_clip
import torch

def load_bioclip():
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
    tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
    return model, tokenizer
    
if __name__ == '__main__':
    model, tokenizer = load_bioclip()
    print(model)
    print(model.transformer)
