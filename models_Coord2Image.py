
from Networks import *
from torch import optim
import torch
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import torchvision.transforms.functional as TF
import random
from utils import eval_redraw
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import lmdb
import shutil
import os



class Sketch_Classification(nn.Module):
    def __init__(self, hp):
        super(Sketch_Classification, self).__init__()
        self.Network = Sketch_LSTM()
        self.hp = hp

        if hp.dataset_name == 'TUBerlin':
            self.hp.num_class = 250
        elif hp.dataset_name == 'QuickDraw':
            self.hp.num_class = 345

        if hp.fullysupervised:

            self.classifier = nn.Linear(512*2, self.hp.num_class)

        else:
            self.embedding = nn.Linear(512*2, 512)
            self.image_decoder = UNet_Decoder()


        self.train_params = self.parameters()
        self.optimizer = optim.Adam(self.train_params, hp.learning_rate)

        self.step = 0

    def train_supervised(self, batch, step):

        self.train()
        self.step = step

        self.optimizer.zero_grad()


        sketch_points = batch['sketch_five_point'].to(device)
        sketch_length = batch['seq_len'].to(device)

        output = self.classifier(self.Network(sketch_points, sketch_length))

        loss = F.cross_entropy(output, batch['sketch_label'].to(device))

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_Coordinate2Image(self, batch, step):

        self.train()
        self.step = step

        self.optimizer.zero_grad()


        sketch_points = batch['sketch_five_point'].to(device)
        sketch_length = batch['seq_len'].to(device)

        feature = self.Network(sketch_points, sketch_length)
        feature = self.embedding(feature)
        reconstructed_image = self.image_decoder(feature)

        loss = F.mse_loss(batch['sketch_img'].to(device), reconstructed_image, reduction='sum')/len(sketch_length)

        loss.backward()
        self.optimizer.step()

        if step % self.hp.draw_frequency == 0:

            batch_redraw = []

            for x, y in zip(batch['sketch_img'], reconstructed_image.detach().cpu()):
                batch_redraw.append(x)
                batch_redraw.append(y)
            batch_redraw = torch.stack(batch_redraw).float()
            torchvision.utils.save_image(batch_redraw,
                                         self.hp.date_time_folder + '/sketch_Viz/' + 'train' + '_' + str(step) + '_.jpg',
                                         normalize=True, nrow=8)

        return loss.item()


    def evaluate_coordinate_redraw(self, batch, step):

        self.eval()
        self.step = step

        sketch_points = batch['sketch_five_point'].to(device)
        sketch_length = batch['seq_len'].to(device)

        feature = self.Network(sketch_points, sketch_length)
        feature = self.embedding(feature)
        reconstructed_image = self.image_decoder(feature)

        batch_redraw = []
        for x, y in zip(batch['sketch_img'], reconstructed_image.detach().cpu()):
            batch_redraw.append(x)
            batch_redraw.append(y)
        batch_redraw = torch.stack(batch_redraw).float()
        torchvision.utils.save_image(batch_redraw,
                                     self.hp.date_time_folder + '/sketch_Viz/' + 'test' + '_' + str(step) + '_.jpg',
                                     normalize=True, nrow=8)


    def fine_tune_linear(self, datloader_Train, datloader_Test):
        self.freeze_weights()

        Train_Image_Feature = {}
        Train_Image_Label = []

        Test_Image_Feature = {}
        Test_Image_Label = []

        start_time = time.time()
        self.eval()

        with torch.no_grad():

            for i_batch, sampled_batch in enumerate(datloader_Train):

                sketch_points = sampled_batch['sketch_five_point'].to(device)
                sketch_length = sampled_batch['seq_len'].to(device)

                feature_A = self.Network(sketch_points, sketch_length)
                feature_B = self.embedding(feature_A)

                sketch_feature = {}

                sketch_feature['BLSTM'] = feature_A
                sketch_feature['Embedding'] = feature_B

                if not Train_Image_Feature:
                    for key in list(sketch_feature.keys()):
                        Train_Image_Feature[key] = []
                for key in list(sketch_feature.keys()):
                    Train_Image_Feature[key].extend(sketch_feature[key].detach())
                Train_Image_Label.extend(sampled_batch['sketch_label'])
                if i_batch%50 == 0:
                    print('Extracting Training Features:' + str(i_batch) + '/' + str(len(datloader_Train)))


            for i_batch, sampled_batch in enumerate(datloader_Test):
                sketch_points = sampled_batch['sketch_five_point'].to(device)
                sketch_length = sampled_batch['seq_len'].to(device)

                feature_A = self.Network(sketch_points, sketch_length)
                feature_B = self.embedding(feature_A)

                sketch_feature = {}

                sketch_feature['BLSTM'] = feature_A
                sketch_feature['Embedding'] = feature_B

                if not Test_Image_Feature:
                    for key in list(sketch_feature.keys()):
                        Test_Image_Feature[key] = []
                for key in list(sketch_feature.keys()):
                    Test_Image_Feature[key].extend(sketch_feature[key].detach())
                Test_Image_Label.extend(sampled_batch['sketch_label'])
                if i_batch%50 == 0:
                    print('Extracting Testing Features: ' + str(i_batch) + '/' + str(len(datloader_Test)))

        Train_Image_Label, Test_Image_Label = torch.tensor(Train_Image_Label).to(device), torch.tensor(Test_Image_Label).to(device)
        save_result = []

        for i_key, key in enumerate(list(Train_Image_Feature.keys())[::-1]):

            Train_Feature, Test_Feature = torch.stack(Train_Image_Feature[key]), torch.stack(Test_Image_Feature[key])
            model = nn.Linear(Train_Feature.shape[1], self.hp.num_class).to(device)
            optimizer = optim.Adam(model.parameters(), 0.0001)

            max_epoch_finetune = [750, 750]
            batch_finetune = 128
            step = 0
            best_accuracy = 0

            for epoch in range(max_epoch_finetune[i_key]):

                for idx in range(len(Train_Feature) // batch_finetune):
                    step = step + 1
                    optimizer.zero_grad()
                    batch_train_feature = Train_Feature[idx * batch_finetune: (idx + 1) * batch_finetune]
                    batch_train_label = Train_Image_Label[idx * batch_finetune: (idx + 1) * batch_finetune]
                    output = model(batch_train_feature)
                    loss = F.cross_entropy(output, batch_train_label)
                    loss.backward()
                    optimizer.step()

                    if step%1000 == 0:
                        prediction = output.argmax(dim=1, keepdim=True)
                        bacth_accuracy = prediction.eq(batch_train_label.view_as(prediction)).sum().item()
                        print('@FineTuning: {}, Epoch: {}, Steps: {}, Iter: {}, Loss: {}, '
                              'Train Accuracy: {}, Max Test Accuracy: {}'.format(key, epoch, step, idx, loss,
                                                                             bacth_accuracy/batch_finetune*100, best_accuracy))

                    if step%500 == 0:
                        output = model(Test_Feature)
                        prediction = output.argmax(dim=1, keepdim=True)
                        test_accuracy = prediction.eq(Test_Image_Label.view_as(prediction)).sum().item()/Test_Feature.shape[0] * 100
                        if test_accuracy > best_accuracy:
                            best_accuracy = test_accuracy

            print("Step: {}::::Layer Name: {} ---> Accuracy: {}".format(self.step, key, best_accuracy))
            save_result.append((key, best_accuracy))

        with open(self.hp.date_time_folder + '/Results.txt', 'a') as filehandle:
            filehandle.write('Step: ' + str(self.step) + '\n')
            np.savetxt(filehandle, np.array(save_result), fmt='%s', comments=str(self.step))

        print('Time to Evaluate:{} Minutes'.format((time.time() - start_time)/60.))

        self.unfreeze_weights()

        return save_result[0][-1]


    def evaluate(self, dataloader_Test):
        self.eval()
        correct = 0
        test_loss = 0
        start_time = time.time()

        with torch.no_grad():

            for i_batch, batch in enumerate(dataloader_Test):

                sketch_points = batch['sketch_five_point'].to(device)
                sketch_length = batch['seq_len'].to(device)

                output = self.classifier(self.Network(sketch_points, sketch_length))

                test_loss += F.cross_entropy(output, batch['sketch_label'].to(device))

                prediction = output.argmax(dim=1, keepdim=True).to('cpu')
                correct += prediction.eq(batch['sketch_label'].view_as(prediction)).sum().item()

            test_loss /= len(dataloader_Test.dataset)
            accuracy = 100. * correct / len(dataloader_Test.dataset)

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Time_Takes: {}\n'.format(
                test_loss, correct, len(dataloader_Test.dataset), accuracy, (time.time() - start_time)))

        return accuracy


    def freeze_weights(self):
        for name, x in self.named_parameters():
            x.requires_grad = False


    def unfreeze_weights(self):
        for name, x in self.named_parameters():
            x.requires_grad = True
