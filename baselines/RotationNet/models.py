
from Networks import *
from torch import optim
import torch
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import torchvision.transforms.functional as TF

class Sketch_Classification(nn.Module):
    def __init__(self, hp):
        super(Sketch_Classification, self).__init__()
        self.Network = eval(hp.backbone_name + '_Network(hp)')
        self.train_params = self.parameters()
        self.optimizer = optim.Adam(self.train_params, hp.learning_rate)
        self.loss = nn.CrossEntropyLoss()
        self.hp = hp
        self.step = 0

    def train_supervised(self, batch, step):
        self.train()
        self.step = step
        self.optimizer.zero_grad()
        output = self.Network(batch['sketch_img'].to(device))

        loss = self.loss(output, batch['sketch_label'].to(device))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_rotation_self_supervised(self, batch, step):

        batch_image_4x = []
        label_4x = []
        for image in batch['sketch_img']:
            batch_image_4x.append(image)
            label_4x.append(0)
            image = TF.normalize(image, mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                    std=[1/0.229, 1/0.224, 1/0.225])
            pil_image = TF.to_pil_image(image)
            tensor_rotate = TF.normalize(TF.to_tensor(TF.rotate(pil_image, 90)),
                                                      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            batch_image_4x.append(tensor_rotate)
            label_4x.append(1)
            tensor_rotate = TF.normalize(TF.to_tensor(TF.rotate(pil_image, 180)),
                                                      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            batch_image_4x.append(tensor_rotate)
            label_4x.append(2)
            tensor_rotate = TF.normalize(TF.to_tensor(TF.rotate(pil_image, 270)),
                                         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            batch_image_4x.append(tensor_rotate)
            label_4x.append(3)

        random_indices = torch.randperm(len(label_4x))
        batch_image_4x = torch.stack(batch_image_4x)[random_indices]
        label_4x = torch.tensor(label_4x)[random_indices]

        self.train()
        self.step = step
        self.optimizer.zero_grad()
        output = self.Network(batch_image_4x.to(device))
        loss = self.loss(output, label_4x.to(device))
        loss.backward()
        self.optimizer.step()
        return loss.item()


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

            max_epoch_finetune = [50, 200, 300, 400]
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

                    if step%100 == 0:
                        prediction = output.argmax(dim=1, keepdim=True)
                        bacth_accuracy = prediction.eq(batch_train_label.view_as(prediction)).sum().item()
                        print('@FineTuning: {}, Epoch: {}, Steps: {}, Iter: {}, Loss: {}, '
                              'Train Accuracy: {}, Max Test Accuracy: {}'.format(key, epoch, step, idx, loss,
                                                                             bacth_accuracy/batch_finetune*100, best_accuracy))

                    if step%1000 == 0:
                        output = model(Test_Feature)
                        prediction = output.argmax(dim=1, keepdim=True)
                        test_accuracy = prediction.eq(Test_Image_Label.view_as(prediction)).sum().item()/Test_Feature.shape[0] * 100
                        if test_accuracy > best_accuracy:
                            best_accuracy = test_accuracy

            print("Step: {}::::Layer Name: {} ---> Accuracy: {}".format(self.step, key, best_accuracy))
            save_result.append((key, best_accuracy))

        with open('Results.txt', 'a') as filehandle:
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
            test_loss += self.loss(output, batch['sketch_label'].to(device)).item()
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



