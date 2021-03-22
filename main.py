import torch
from models_Image2Coord import *
# from models_Coord2Image import *
from Dataset import get_dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse
import os
from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y %H:%M:%S")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Skecth_Classification')
    parser.add_argument('--dataset_name', type=str, default='QuickDraw', help='TUBerlin vs QuickDraw')
    parser.add_argument('--fullysupervised', type=bool, default=True) #Change here -- be careful



    parser.add_argument('--backbone_name', type=str, default='Resnet', help='VGG/Resnet')
    parser.add_argument('--pool_method', type=str, default='AdaptiveAvgPool2d')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--nThreads', type=int, default=8)
    parser.add_argument('--splitTrain', type=float, default=0.9)

    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--eval_freq_iter', type=int, default=48000)
    parser.add_argument('--print_freq_iter', type=int, default=50)
    parser.add_argument('--draw_frequency', type=int, default=100)
    parser.add_argument('--drop_rate', type=float, default=0.5)


    parser.add_argument('--encoder_hidden_dim', type=int, default=128)
    parser.add_argument('--decoder_hidden_dim', type=int, default=512)


    hp = parser.parse_args()
    dataloader_Train, dataloader_Test = get_dataloader(hp)
    print(hp)
    hp.date_time_folder = os.path.join('./results/', hp.dataset_name, dt_string)
    if not os.path.exists(hp.date_time_folder):
        os.makedirs(hp.date_time_folder)
        os.makedirs(hp.date_time_folder + '/sketch_Viz')
        os.makedirs(hp.date_time_folder + '/models')

    #######################################################################
    ############################## End Load Data ##########################


    model = Sketch_Classification(hp)
    model.to(device)


    step = 0
    best_accuracy = 0


    # # ####################### Self-Supervised Training Image2coordinate Net #########################
    # # ###########################################################################
    # for epoch in range(hp.max_epoch):
    #     for i_batch, batch in enumerate(dataloader_Train):
    #         loss = model.train_Image2Coordinate(batch, step)
    #         step += 1
    #         if (step + 0) % hp.print_freq_iter == 0:
    #             print('Epoch: {}, Iter: {}, Steps: {}, '
    #                   'Loss: {}, Best Accuracy: {}'.format(epoch, i_batch, step, loss, best_accuracy))
    #
    #
    #         if step%20000 == 0:
    #             torch.save(model.state_dict(), hp.date_time_folder + '/models/Image2coordinate' + str(step) +  '.pth')
    #
    #         if (step + 1)  % hp.draw_frequency == 0:
    #             with torch.no_grad():
    #                 model.evaluate_coordinate_redraw(batch, step)
    #
    #         if (step + 0) % hp.eval_freq_iter == 0:
    #             accuracy = model.fine_tune_linear_LMDB(dataloader_Train, dataloader_Test)
    #             if accuracy > best_accuracy:
    #                 best_accuracy = accuracy
    #                 torch.save(model.state_dict(), hp.date_time_folder + '/models/Image2coordinate.pth')



    ####################### Fully Supervised Training #########################
    ###########################################################################
    for epoch in range(hp.max_epoch):
        for i_batch, batch in enumerate(dataloader_Train):
            loss = model.train_supervised(batch, step)
            step += 1
            if (step + 0) % hp.print_freq_iter == 0:
                print('Epoch: {}, Iter: {}, Steps: {}, '
                      'Loss: {}, Best Accuracy: {}'.format(epoch, i_batch, step, loss, best_accuracy))
            if (step + 0) % hp.eval_freq_iter == 0:
                with torch.no_grad():
                    accuracy = model.evaluate(dataloader_Test)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), hp.date_time_folder + '/models/modelSupervised.pth')
            if step%20000 == 0:
                torch.save(model.state_dict(), hp.date_time_folder + '/models/modelSupervised' + str(step) +  '.pth')


    # ###################### Self-Supervised Training Coordinate2Image Net #########################
    # ##########################################################################
    # for epoch in range(hp.max_epoch):
    #     for i_batch, batch in enumerate(dataloader_Train):
    #         loss = model.train_Coordinate2Image(batch, step)
    #         step += 1
    #         if (step + 0) % hp.print_freq_iter == 0:
    #             print('Epoch: {}, Iter: {}, Steps: {}, '
    #                   'Loss: {}, Best Accuracy: {}'.format(epoch, i_batch, step, loss, best_accuracy))
    #
    #         if step%20000 == 0:
    #             torch.save(model.state_dict(), hp.date_time_folder + '/models/Coordinate2Image' + str(step) +  '.pth')
    #
    #         if (step + 1)  % hp.draw_frequency == 0:
    #             with torch.no_grad():
    #                 model.evaluate_coordinate_redraw(batch, step)
    #
    #         if (step + 0) % hp.eval_freq_iter == 0:
    #             accuracy = model.fine_tune_linear(dataloader_Train, dataloader_Test)
    #             if accuracy > best_accuracy:
    #                 best_accuracy = accuracy
    #                 torch.save(model.state_dict(), hp.date_time_folder + '/models/Image2coordinate.pth')
