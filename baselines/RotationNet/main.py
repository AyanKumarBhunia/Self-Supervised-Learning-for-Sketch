import torch
from models import *
from dataset import get_dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Skecth_Classification')
    parser.add_argument('--backbone_name', type=str, default='Resnet', help='VGG/Resnet')
    parser.add_argument('--pool_method', type=str, default='AdaptiveAvgPool2d')
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--nThreads', type=int, default=8)
    parser.add_argument('--splitTrain', type=float, default=0.9)

    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=60)
    parser.add_argument('--eval_freq_iter', type=int, default=1000)
    parser.add_argument('--print_freq_iter', type=int, default=50)
    parser.add_argument('--drop_rate', type=float, default=0.5)
    parser.add_argument('--supervised', type=bool, default=True)


    hp = parser.parse_args()
    dataloader_Train, dataloader_Test = get_dataloader(hp)
    print(hp)

    #######################################################################
    ############################## End Load Data ##########################
    model = Sketch_Classification(hp)
    model.to(device)

    # model.fine_tune_linear(dataloader_Train, dataloader_Test)
    # model.load_state_dict(torch.load('/home/media/CVPR_2021/Sketch_SelfSupervised/SSL_Sketch/models/SSL_Rotation/model_best_rotationet.pth', map_location=device))
    # model.fine_tune_linear(dataloader_Train, dataloader_Test)
    # model.evaluate(dataloader_Test)
    step = 0
    best_accuracy = 0

    ####################### Self-Supervised Training #########################
    ###########################################################################
    for epoch in range(hp.max_epoch):
        for i_batch, batch in enumerate(dataloader_Train):
            loss = model.train_rotation_self_supervised(batch, step)
            step += 1
            if (step + 0) % hp.print_freq_iter == 0:
                print('Epoch: {}, Iter: {}, Steps: {}, '
                      'Loss: {}, Best Accuracy: {}'.format(epoch, i_batch, step, loss, best_accuracy))
            if (step + 1) % hp.eval_freq_iter == 0:
                accuracy = model.fine_tune_linear(dataloader_Train, dataloader_Test)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), 'model_best_rotationet.pth')

    # ####################### Fully Supervised Training #########################
    # ###########################################################################
    # for epoch in range(hp.max_epoch):
    #     for i_batch, batch in enumerate(dataloader_Train):
    #         loss = model.train_supervised(batch, step)
    #         step += 1
    #         if (step + 0) % hp.print_freq_iter == 0:
    #             print('Epoch: {}, Iter: {}, Steps: {}, '
    #                   'Loss: {}, Best Accuracy: {}'.format(epoch, i_batch, step, loss, best_accuracy))
    #         if (step + 1) % hp.eval_freq_iter == 0:
    #             accuracy = model.evaluate(dataloader_Test)
    #             if accuracy > best_accuracy:
    #                 best_accuracy = accuracy
    #                 torch.save(model.state_dict(), 'model_best.pth')
