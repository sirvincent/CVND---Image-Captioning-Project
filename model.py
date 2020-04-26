import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        
        self.linear = nn.Linear(hidden_size, vocab_size)

        
    
    def forward(self, features, captions):
        embeddings = self.embedding(captions[:, :-1])
        features = features.unsqueeze(1)
        
        embedded_input = torch.cat((features, embeddings), 1)
        
        # TODO: is the hidden state necessary? it is unused currently so no need to assign a variable for it
        lstm_output, _ = self.lstm(embedded_input)
        
        output = self.linear(lstm_output)
        
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        predicted_word_tokens = []
        for idx in range(max_len):
            predictions, states = self.lstm(inputs, states)
            predictions = self.linear(predictions)
            
            predicted_word_token = predictions.argmax()
            
            # TODO: magic number 1, pas end token as parameter
            if predicted_word_token == 1:
                # end word has been found
                break
                
            predicted_word_tokens.append(predicted_word_token.item())

            # feed latest state into new state of lstm      
            inputs = self.embedding(predicted_word_token)   
            inputs = inputs.unsqueeze(0).unsqueeze(0)
        
        return predicted_word_tokens
            
            
        