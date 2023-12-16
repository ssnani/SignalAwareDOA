import torch
import torch.nn as nn
import torch.nn.functional as F
from debugging_network import *

#
DROPOUT_FLAG = True    #TODO:


class ADNN(nn.Module):
    def __init__(self, hidden_size, bidirectional=False):
        super(ADNN, self).__init__()

        hid_out = hidden_size
        self.lstm_1 = nn.LSTM(257, hid_out, 1, batch_first=True, bidirectional=bidirectional)
        self.dropout_1 = nn.Dropout(0.4)
        self.lstm_2 = nn.LSTM(hid_out, hid_out, 1, batch_first=True, bidirectional=bidirectional)
        self.dropout_2 = nn.Dropout(0.7)
        self.linear = nn.Linear(512,257)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x: "[batch_size, num_frames, num_freq]"):

        out_1, _ = self.lstm_1(x)
        if DROPOUT_FLAG:
            out_1 = self.dropout_1(out_1)

        out_2, _ = self.lstm_2(out_1)
        if DROPOUT_FLAG:
            out_2 = self.dropout_2(out_2)

        out_3 = self.linear(out_2)

        out_mask = self.sigmoid(out_3)

        return out_mask


class DDNN(nn.Module):

    def __init__(self, num_mics=4, num_freq=257, num_doa_classes=37):
        super(DDNN, self).__init__()
        self.num_mics = num_mics
        self.num_freq = num_freq
        self.num_doa_classes = num_doa_classes

        self.padding = 0
        self.kernel_size = (2,1)   #(num_mics, num_freq) -> multi_spk: (2,1), single_spk: (2,2)
        self.stride = 1
        self.num_conv_out_channels = 64
        self.fc_out_dim = 512


        self.conv1 = nn.Sequential(nn.Conv2d(1, self.num_conv_out_channels, self.kernel_size, padding = self.padding, stride=self.stride),
                                    nn.BatchNorm2d(self.num_conv_out_channels),
                                    nn.ReLU())
        self.conv_layers = nn.ModuleList([ nn.Sequential(nn.Conv2d(self.num_conv_out_channels, self.num_conv_out_channels, self.kernel_size, padding = self.padding, stride=self.stride), 
                                            nn.BatchNorm2d(self.num_conv_out_channels),
                                            nn.ReLU()) for i in range(1, self.num_mics-2)
                                            ])

        self.conv_final_layer =  nn.Sequential(nn.Conv2d(self.num_conv_out_channels, self.num_conv_out_channels, self.kernel_size, padding = self.padding, stride=self.stride), 
                                            nn.ReLU())
                                            
        self.fc = nn.Linear(self.num_conv_out_channels*self.num_freq, self.fc_out_dim)
        self.fc_2 = nn.Linear(self.fc_out_dim, self.fc_out_dim)
        self.doa_layer = nn.Linear(self.fc_out_dim, self.num_doa_classes)
        self.relu = nn.ReLU()
        self.dropout_1 = nn.Dropout(p=0.5)
        self.dropout_2 = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: "[batch_size, num_frames, num_mics, num_freq]", mask: "[batch_size, num_frames, num_freq]"):

        batch_size, num_frames, num_mics, num_freq = input.shape

        input = torch.unsqueeze(input, dim=2)
        input_frames = torch.reshape(input, (batch_size*num_frames, 1, num_mics, num_freq))

        out1 = self.conv1(input_frames)
        
        for lyr in self.conv_layers:
            out1 = lyr(out1)

        out1 = self.conv_final_layer(out1)

        #[batch_size*num_frames, 64, 1, num_freq]
        
        out1 = torch.reshape(torch.squeeze(out1), (batch_size, num_frames, 64, num_freq))
        mask = torch.unsqueeze(mask, dim=2)
        mask_out = out1*mask

        fc_input = torch.reshape(mask_out, (batch_size, num_frames, -1))

        fc_out = self.relu(self.fc(fc_input))
        if DROPOUT_FLAG:
            fc_out = self.dropout_1(fc_out)

        fc_out_2 = self.relu(self.fc_2(fc_out))
        if DROPOUT_FLAG:
            fc_out_2 = self.dropout_2(fc_out_2)

        out_num_doa_classess = self.sigmoid(self.doa_layer(fc_out_2)) #  #TODO:

        return out_num_doa_classess


class SignalAwareDoA(nn.Module):
    def __init__(self, hidden_size=512, bidirectional=False, num_mics=4, num_freq=257, num_doa_classes=37):
        super(SignalAwareDoA, self).__init__()
        self.adnn = ADNN(hidden_size, bidirectional)
        #get_all_layers(self.adnn)
        self.ddnn = DDNN( num_mics, num_freq, num_doa_classes)
        #get_all_layers(self.ddnn)

    def forward(self, mix_mag_ch: "[batch_size, num_frames, num_freq]", mix_ph_all: "[batch_size, num_frames, num_mics, num_freq]"):
        mask = self.adnn(mix_mag_ch)
        doa_class = self.ddnn(mix_ph_all, mask)
        return doa_class



if __name__ =="__main__":
    adnn = ADNN(512, False)

    input = torch.randn(16,100,257)
    output_mask = adnn(input)
    print(output_mask.shape)

    ddnn = DDNN()

    input_ph = torch.randn(16,100,4,257)
    out_num_doa_classess = ddnn(input_ph, output_mask)
    print(out_num_doa_classess.shape)

    sa_dnn = SignalAwareDoA()
    out_num_doa = sa_dnn(input, input_ph)
    print(out_num_doa.shape)




