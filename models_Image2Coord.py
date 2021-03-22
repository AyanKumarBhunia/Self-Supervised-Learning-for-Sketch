
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
        self.Network = eval(hp.backbone_name + '_Network(hp)')

        if not hp.fullysupervised:
            self.fc_encoded_feature = nn.Linear(2048,  hp.encoder_hidden_dim)
            # self.fc_hidden = nn.Linear(2048, self.decoder_hidden_dim)
            self.decoder = nn.GRU(5 + hp.encoder_hidden_dim, hp.decoder_hidden_dim, batch_first=True)
            self.linear_output = nn.Linear(hp.decoder_hidden_dim, 5)

        self.train_params = self.parameters()
        self.optimizer = optim.Adam(self.train_params, hp.learning_rate)
        self.hp = hp
        self.step = 0


    def train_supervised(self, batch, step):

        self.train()
        self.step = step

        self.optimizer.zero_grad()
        output = self.Network(batch['sketch_img'].to(device))

        loss = F.cross_entropy(output, batch['sketch_label'].to(device))

        loss.backward()
        self.optimizer.step()
        return loss.item()


    def train_Image2Coordinate(self, batch, step, teacher_forcing_ratio = 0.5):

        self.train()
        self.step = step
        self.optimizer.zero_grad()

        target_coord = batch['sketch_five_point'].float().to(device)
        # target_coord[:, :, :2] = target_coord[:, :, :2]/256. #No need to Nornamize/ unseen samples poor recocntruction

        x = self.Network(batch['sketch_img'].to(device))
        # self.hidden = F.tanh(self.fc_hidden(x)).unsqueeze(0)
        encoded_feature = self.fc_encoded_feature(x)
        use_teacher_forcing = False # if random.random() < teacher_forcing_ratio else False


        if use_teacher_forcing:
            loss = 0
            batch_output = []
            hidden = None

            for i_num in range(batch['seq_len'].max()):

                decoder_input = torch.cat((encoded_feature.unsqueeze(1), target_coord[:, i_num, :].unsqueeze(1)), dim=-1)
                _, hidden = self.decoder(decoder_input, hidden)
                output = self.linear_output(hidden[0])

                output_XY, pen_bits = output.split([2, 3], dim=1)
                # output_XY = torch.sigmoid(output_XY)

                pen_bits_onehot = F.one_hot(pen_bits.argmax(-1), num_classes=3).float()
                batch_output.append(torch.cat((output_XY.detach(), pen_bits_onehot), dim=-1))

                mask = 1. - target_coord[:, i_num + 1, 4]
                loss_coor = F.mse_loss(output_XY, target_coord[:, i_num + 1, :2], reduction='none') * mask.unsqueeze(-1)
                loss_pen = F.cross_entropy(pen_bits, target_coord[:, i_num + 1, 2:].argmax(axis=-1),
                                           reduction='none') * mask

                loss_coor = loss_coor.sum()
                loss_pen = loss_pen.sum()

                loss += loss_coor + loss_pen

            loss = loss / batch['sketch_img'].shape[0]
            batch_output = torch.stack(batch_output, dim=1)

        else:

            decoder_input = torch.cat((encoded_feature.unsqueeze(1).repeat(1, batch['seq_len'].max(), 1), target_coord[:,:-1, :]), dim=-1)
            decoder_input = pack_padded_sequence(decoder_input, batch['seq_len'],  batch_first=True, enforce_sorted=False)
            output_hiddens, _ = self.decoder(decoder_input)
            output_hiddens, _ = pad_packed_sequence(output_hiddens, batch_first=True)

            output = self.linear_output(output_hiddens)

            output_XY, pen_bits = output.split([2, 3], dim=-1)
            # output_XY = torch.sigmoid(output_XY)

            pen_bits_onehot = F.one_hot(pen_bits.argmax(-1), num_classes=3).float()
            batch_output = torch.cat((torch.clamp(output_XY.detach(), min=0, max=256.0) , pen_bits_onehot), dim=-1)

            mask = torch.ones(output.shape[:2]).to(device)
            target_cross_entropy = target_coord[:, 1:, 2:].argmax(axis=-1)
            for i_num, seq in enumerate(batch['seq_len']):
                mask[i_num, seq:] = 0.
                target_cross_entropy[i_num, seq:] = 10

            loss_coor = F.mse_loss(output_XY, target_coord[:, 1:, :2], reduction='none') * mask.unsqueeze(-1)

            # loss_pen = F.cross_entropy(pen_bits.view(-1, pen_bits.shape[-1]), target_cross_entropy.view(-1), ignore_index=10,
            #                            reduction='none') * mask.view(-1)
            loss_pen = F.cross_entropy(pen_bits.view(-1, pen_bits.shape[-1]), target_cross_entropy.view(-1), ignore_index=10)

            # loss = (100*loss_coor.sum() + loss_pen.sum()) / batch['sketch_img'].shape[0]

            loss_coor = loss_coor.sum() / mask.sum()

            loss = loss_coor + loss_pen


        loss.backward()
        self.optimizer.step()

        if self.step % self.hp.draw_frequency == 0:
            eval_redraw(target_coord[:, 1:, :], batch_output,  batch['seq_len'], self.step, self.hp.date_time_folder, type = 'Train')

        # if self.step%10 == 0:
        #     print('Loss Coordinate: {}, Loss Pen Bits: {}'.format(loss_coor.item(),  loss_pen.item()))

        return loss.item()



    def evaluate_coordinate_redraw(self, batch, step):

        self.eval()

        target_coord = batch['sketch_five_point'].float().to(device)
        # target_coord[:, :, :2] = target_coord[:, :, :2] / 256.

        x = self.Network(batch['sketch_img'].to(device))
        # hidden = F.tanh(self.fc_hidden(x)).unsqueeze(0)
        encoded_feature = self.fc_encoded_feature(x)

        batch_output = []
        targets = torch.zeros(batch['sketch_img'].shape[0], 5).to(device)  # [GO] token
        hidden = None

        for i_num in range(batch['seq_len'].max()):
            decoder_input = torch.cat((encoded_feature.unsqueeze(1), targets.unsqueeze(1)), dim=-1)
            _, hidden = self.decoder(decoder_input, hidden)
            output = self.linear_output(hidden[0])

            output_XY, pen_bits = output.split([2, 3], dim=1)
            # output_XY = torch.sigmoid(output_XY)

            # pen_bits_onehot = F.one_hot(pen_bits.argmax(-1), num_classes=3).float()
            pen_bits_onehot = target_coord[:, i_num+1, 2:]
            # targets = target_coord[:, i_num+1, :]
            targets = torch.cat((torch.clamp(output_XY.detach(), min=0, max=256.0), pen_bits_onehot), dim=-1)
            batch_output.append(targets)

        batch_output = torch.stack(batch_output, dim=1)
        eval_redraw(target_coord[:, 1:, :], batch_output, batch['seq_len'], step, self.hp.date_time_folder, type = 'Eval')



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
                sketch_feature = self.Network.extract_features(sampled_batch['sketch_img'].to(device))
                if not Train_Image_Feature:
                    for key in list(sketch_feature.keys()):
                        Train_Image_Feature[key] = []
                for key in list(sketch_feature.keys()):
                    Train_Image_Feature[key].extend(sketch_feature[key].detach())
                Train_Image_Label.extend(sampled_batch['sketch_label'])
                if i_batch%50 == 0:
                    print('Extracting Training Features:' + str(i_batch) + '/' + str(len(datloader_Train)))


            for i_batch, sampled_batch in enumerate(datloader_Test):
                sketch_feature = self.Network.extract_features(sampled_batch['sketch_img'].to(device))
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
            model = nn.Linear(Train_Feature.shape[1], 250).to(device)
            optimizer = optim.Adam(model.parameters(), 0.0001)

            max_epoch_finetune = [750, 750, 750, 750]
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



    def fine_tune_linear_LMDB(self, datloader_Train, datloader_Test):

        self.freeze_weights()
        if os.path.exists('./feature_extracted'):
            shutil.rmtree('./feature_extracted')

        Data_ENV = lmdb.open('./feature_extracted', map_size=10e10)

        Train_Image_Feature = {}
        Train_Image_Label = []

        Test_Image_Feature = {}
        Test_Image_Label = []

        Train_Key, Test_Key = [], []
        key_feature_len = {}

        start_time = time.time()
        self.eval()



        with torch.no_grad():

            count = -1
            for i_batch, sampled_batch in enumerate(datloader_Train):
                if i_batch%20 == 0:
                    print('Extracting Training Features:' + str(i_batch) + '/' + str(len(datloader_Train)))

                sketch_feature = self.Network.extract_features(sampled_batch['sketch_img'].to(device))

                if not Train_Image_Feature:
                    for key in list(sketch_feature.keys()):
                        Train_Image_Feature[key] = []


                for key in list(sketch_feature.keys()):
                    Train_Image_Feature[key].extend(sketch_feature[key].detach().cpu().numpy())

                Train_Image_Label.extend(sampled_batch['sketch_label'].numpy())

                if ((i_batch+1)%100 == 0) or (i_batch == len(datloader_Train)-1):

                    print('Extracting Training Features:' + str(i_batch) + '/' + str(len(datloader_Train)))
                    count = count + 1
                    with Data_ENV.begin(write=True) as txn:

                        for key in list(sketch_feature.keys()):

                            # print(np.array(Train_Image_Feature[key]).shape)

                            data = np.array(Train_Image_Feature[key]).tobytes()
                            key_lmdb_Feature = key + '_Train_Feature_' + str(count)
                            txn.put(key_lmdb_Feature.encode("ascii"), data)

                            data = np.array(Train_Image_Label).tobytes()
                            key_lmdb_Label = key + '_Train_Label_' + str(count)
                            txn.put(key_lmdb_Label.encode("ascii"), data)

                            Train_Key.append([key_lmdb_Feature, key_lmdb_Label])

                        Train_Image_Feature = {}
                        Train_Image_Label = []

            max_Train_count = count

            count = -1
            for i_batch, sampled_batch in enumerate(datloader_Test):

                sketch_feature = self.Network.extract_features(sampled_batch['sketch_img'].to(device))

                if not Test_Image_Feature:
                    for key in list(sketch_feature.keys()):
                        Test_Image_Feature[key] = []
                        key_feature_len[key] = sketch_feature[key].shape[-1]

                for key in list(sketch_feature.keys()):
                    Test_Image_Feature[key].extend(sketch_feature[key].detach().cpu().numpy())

                Test_Image_Label.extend(sampled_batch['sketch_label'])

                if ((i_batch+1)%100 == 0) or (i_batch == len(datloader_Test)-1):

                    print('Extracting Training Features:' + str(i_batch) + '/' + str(len(datloader_Test)))
                    count = count + 1

                    with Data_ENV.begin(write=True) as txn:
                        for key in list(sketch_feature.keys()):

                            data = np.array(Test_Image_Feature[key]).tobytes()
                            key_lmdb_Feature = key + '_Test_Feature_' + str(count)
                            txn.put(key_lmdb_Feature.encode("ascii"), data)

                            data = np.array(Test_Image_Label).tobytes()
                            key_lmdb_Label = key + '_Test_Label_' + str(count)
                            txn.put(key_lmdb_Label.encode("ascii"), data)

                            Test_Key.append([key_lmdb_Feature, key_lmdb_Label])

                        Test_Image_Feature = {}
                        Test_Image_Label = []

            max_Test_count = count

        print('Time required to extract the feature - {} Minutes'.format((time.time() - start_time)/60))

        save_result = []
        Data_ENV = lmdb.open('./feature_extracted', max_readers=1,
                  readonly=True, lock=False, readahead=False, meminit=False)

        with Data_ENV.begin(write=False) as txn:

            for i_key, key in enumerate(list(key_feature_len.keys())[::-1]):

                model = nn.Linear(key_feature_len[key], 250).to(device)
                optimizer = optim.Adam(model.parameters(), 0.0001)

                max_epoch_finetune = [750, 750, 750, 750]
                batch_finetune = 128
                step = 0
                best_accuracy = 0

                for epoch in range(max_epoch_finetune[i_key]):

                    for train_count in range(max_Train_count+1):

                        key_lmdb_Feature = key + '_Train_Feature_' + str(train_count)
                        key_lmdb_Label = key + '_Train_Label_' + str(train_count)

                        feature_chunk = txn.get(key_lmdb_Feature.encode("ascii"))
                        label_chunk = txn.get(key_lmdb_Label.encode("ascii"))
                        feature_chunk = torch.from_numpy(np.frombuffer(feature_chunk, dtype=np.float32).reshape(-1, key_feature_len[key]).copy()).to(device)
                        label_chunk = torch.from_numpy(np.frombuffer(label_chunk, dtype = np.int64).copy()).to(device)

                        for idx in range(len(feature_chunk) // batch_finetune):
                            step = step + 1
                            print(step)
                            optimizer.zero_grad()
                            batch_train_feature = feature_chunk[idx * batch_finetune: (idx + 1) * batch_finetune]
                            batch_train_label = label_chunk[idx * batch_finetune: (idx + 1) * batch_finetune]
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

                            if step%10000 == 0:

                                Total_correct = 0
                                Total_sample = 0

                                for test_count in range(max_Test_count+1):
                                    key_lmdb_Feature = key + '_Test_Feature_' + str(test_count)
                                    key_lmdb_Label = key + '_Test_Label_' + str(test_count)

                                    Test_Feature = txn.get(key_lmdb_Feature.encode("ascii"))
                                    Test_Image_Label = txn.get(key_lmdb_Label.encode("ascii"))
                                    Test_Feature = torch.from_numpy(np.frombuffer(Test_Feature, dtype=np.float32).reshape(-1, key_feature_len[key]).copy()).to(device)
                                    Test_Image_Label = torch.from_numpy(np.frombuffer(Test_Image_Label, dtype = np.int64).copy()).to(device)

                                    output = model(Test_Feature)
                                    prediction = output.argmax(dim=1, keepdim=True)

                                    Total_correct +=  prediction.eq(Test_Image_Label.view_as(prediction)).sum().item()
                                    Total_sample += Test_Feature.shape[0]

                                test_accuracy = Total_correct/Total_sample * 100
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
        for i_batch, batch in enumerate(dataloader_Test):
            output = self.Network(batch['sketch_img'].to(device))
            test_loss += F.cross_entropy(output, batch['sketch_label'].to(device)).item()
            prediction = output.argmax(dim=1, keepdim=True).to('cpu')
            correct += prediction.eq(batch['sketch_label'].view_as(prediction)).sum().item()

        test_loss /= len(dataloader_Test.dataset)
        accuracy = 100. * correct / len(dataloader_Test.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Time_Takes: {}\n'.format(
            test_loss, correct, len(dataloader_Test.dataset), accuracy, (time.time() - start_time) ))

        return accuracy


    def freeze_weights(self):
        for name, x in self.named_parameters():
            x.requires_grad = False


    def unfreeze_weights(self):
        for name, x in self.named_parameters():
            x.requires_grad = True



