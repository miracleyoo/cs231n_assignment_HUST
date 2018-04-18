# coding: utf-8
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import json
import datetime
import numpy as np
import visdom

from torch.autograd import Variable
from tqdm import tqdm
from config import Config

def training(train_loader, test_loader, weights, top_num=1):

    print('==> Loading Model ...')
    opt       = Config()

    NUM_TRAIN = len(train_loader)*opt.BATCH_SIZE
    NUM_TEST  = len(test_loader) *opt.BATCH_SIZE
    NUM_TRAIN_PER_EPOCH = len(train_loader)
    NUM_TEST_PER_EPOCH  = len(test_loader )

    criterion = nn.BCEWithLogitsLoss(weight=weights,size_average=False)
    net       = opt.MODEL

    try:
        net   = torch.load(opt.NET_SAVE_PATH+'%s_model_temp.pkl'%(net.__class__.__name__))
        print("Load existing model: %s"%(opt.NET_SAVE_PATH+'%s_model_temp.pkl'%(net.__class__.__name__)))
    except:
        pass
    if opt.USE_CUDA:net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=opt.LEARNING_RATE)

    train_recorder= {'loss':[],'acc':[],'epoch_loss':[],'epoch_acc':[]}
    test_recorder = {'loss':[],'acc':[]}

    for epoch in range(opt.NUM_EPOCHS):
        running_loss = 0
        running_acc  = 0
        test_loss    = 0
        test_acc     = 0
        train_loss   = 0
        train_acc    = 0
        best_test_acc= 0

        # Start training
        net.train()
        print('==> Preparing Data ...')
        for i, data in tqdm(enumerate(train_loader), desc="Training", total=NUM_TRAIN_PER_EPOCH, leave=False, unit='b'):
            
            inputs, labels, *_ = data
            if opt.USE_CUDA:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Do statistics for training
            train_loss   += loss.data[0]
            running_loss += loss.data[0]
            _, predicts   = torch.max(outputs, 1)
            predicts      = predicts.data
            num_correct   = 0

            if opt.USE_CUDA:
                labels_data   = labels.cpu().data.numpy()
            else:
                labels_data   = labels.data.numpy()

            for i, predict in enumerate(predicts):
                if predict in list(np.where(labels_data[i]==1)[0]):
                    num_correct += 1
            train_acc    += num_correct
            running_acc  += num_correct
            train_recorder['loss'].append(loss.data[0])
            train_recorder['acc' ].append(num_correct)


            # print statistics for each batch
            if opt.PRINT_BATCH and ((i+1)%opt.NUM_PRINT_BATCH == 0):
                print('Batch/Epoch [%d/%d], Train Loss: %.4f, Train Acc: %.4f, Correct Num: %d'
                      %(i+1, epoch+1, running_loss/(opt.BATCH_SIZE*opt.NUM_PRINT_BATCH),\
                       num_correct.data[0]/opt.BATCH_SIZE, num_correct))
                running_loss = 0; running_acc = 0
            
        # Save a temp model
        torch.save(net, opt.NET_SAVE_PATH+'%s_model_temp.pkl'%(net.__class__.__name__))

        # Start testing
        net.eval()
        for i, data in tqdm(enumerate(test_loader), desc="Testing", total=NUM_TEST_PER_EPOCH, leave=False, unit='b'):
            inputs, labels, *_ = data
            if opt.USE_CUDA:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
                
            # Compute the outputs and judge correct
            outputs     = net(inputs)
            loss        = criterion(outputs, labels)
            predicts    = torch.sort(outputs,descending=True)[1][:,:top_num]
            predicts    = predicts.data
            num_correct = 0

            if opt.USE_CUDA:
                labels_data   = labels.cpu().data.numpy()
            else:
                labels_data   = labels.data.numpy()
                
            for i, predict in enumerate(predicts):
                for label in predict:
                    if label in list(np.where(labels_data[i]==1)[0]):
                        num_correct += 1
                        break

            # Do statistics for training
            test_loss  += loss.data[0]
            test_acc   += num_correct

        # Do recording for each epoch
        train_recorder['epoch_loss'].append(train_loss / NUM_TRAIN)
        train_recorder['epoch_acc' ].append(train_acc  / NUM_TRAIN)
        test_recorder['loss'].append(test_loss / NUM_TEST)
        test_recorder['acc' ].append(test_acc  / NUM_TEST)

        # Write log to files
        t = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log_file_name = "%s_epoch_%d_%s.txt"%(net.__class__.__name__, epoch, t)
        with open('./source/log/'+log_file_name, 'w+') as fp:
            json.dump({'train_recorder':train_recorder,'test_recorder':test_recorder}, fp)

        # Output results
        print ('Epoch [%d/%d], Train Loss: %.4f, Train Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f' 
                        %(epoch+1, opt.NUM_EPOCHS, 
                          train_loss / NUM_TRAIN, train_acc / NUM_TRAIN, 
                          test_loss / NUM_TEST, test_acc / NUM_TEST))
        if (test_acc / NUM_TEST) > best_test_acc:
            best_test_acc = test_acc / NUM_TEST
            torch.save(net, opt.NET_SAVE_PATH+'%s_model.pkl'%(net.__class__.__name__))
            
    print('==> Training Finished.')
    return net

def validating(val_loader, net, weights, top_num=1):
    print('==> Loading Model ...')
    opt       = Config()
    val_loss  = 0
    val_acc   = 0 
    NUM_VAL = len(val_loader)*opt.BATCH_SIZE
    NUM_VAL_PER_EPOCH = len(val_loader)

    criterion = nn.BCEWithLogitsLoss(weight=weights,size_average=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.LEARNING_RATE)
    if opt.USE_CUDA:net.cuda()

    net.eval()
    for i, data in tqdm(enumerate(val_loader), desc="Testing", total=NUM_VAL_PER_EPOCH, leave=False, unit='b'):
        inputs, labels, *_ = data
        if opt.USE_CUDA:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
            
        # Compute the outputs and judge correct
        outputs     = net(inputs)
        loss        = criterion(outputs, labels)
        predicts    = torch.sort(outputs,descending=True)[1][:,:top_num]
        predicts    = predicts.data
        num_correct = 0

        if opt.USE_CUDA:
            labels_data   = labels.cpu().data.numpy()
        else:
            labels_data   = labels.data.numpy()
            
        for i, predict in enumerate(predicts):
            for label in predict:
                if label in list(np.where(labels_data[i]==1)[0]):
                    num_correct += 1
                    break

        # Do statistics for training
        val_loss  += loss.data[0]
        val_acc   += num_correct

    # Output results
    t = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    val_file_name = "%s_val_%s.txt"%(net.__class__.__name__, t)
    val_results   = "Validation Loss:%.4f, Validation Acc:%.4f"%(val_loss/NUM_VAL, val_acc/NUM_VAL)
    with open('./source/val_results/'+val_file_name, 'w+') as fp:
        fp.writelines(val_results)
    print(val_results)