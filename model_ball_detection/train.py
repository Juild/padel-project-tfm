import random
import sys
import time
import numpy as np
sys.path.append('..')
from torch import Tensor
import config
from model import BallClassifier
import dataset as ds
from torch.utils.data import DataLoader
import torch
import torcheval

from torcheval.metrics import BinaryAUPRC, BinaryPrecisionRecallCurve, BinaryConfusionMatrix
from torchvision import transforms
from torchvision.models import resnet50

import matplotlib.pyplot as plt
from utilities.preprocessing import get_channel_mean_std
from utilities.utils import import_data
import os
import argparse
import sys
import json
import pickle
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
sys.path.append("..")


def save_plot_metrics(train_losses, epochs, batches):
    plt.figure()
    plt.style.use('dark_background')
    plt.xlabel('EPOCHS')
    plt.ylabel('Training loss')
    for train_loss, epochs, batches in zip(train_losses, epochs, batches):
        plt.plot(list(range(epochs)), train_loss, '-o', label=str(epochs))
    plt.legend()
    plt.savefig('../figures/training_loss_history.png')
    plt.close()

def train_model(train_loader, test_loader, loss_func, learning_rate, EPOCHS, eval):
    resnet = resnet50(weights='DEFAULT')
    for param in resnet.parameters():
        param.requires_grad = True
    model = BallClassifier(base_model=resnet, num_classes=2).to(config.DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss = []
    test_losses = []
    train_roc_thrs = []
    test_roc_thrs = []
    train_pr_thrs = []
    test_pr_thrs = []
    for epoch in range(EPOCHS):
        if epoch == 0:
            start = time.time()
        print(f'Epoch: {epoch}')
        model.train()
        loss = 0
        for (images, labels) in train_loader:
            images, labels = images.to(config.DEVICE, dtype=torch.float), labels.type(
                torch.LongTensor).to(config.DEVICE)
            opt.zero_grad()
            predicted_labels: Tensor = model(images)
            total_loss: Tensor = loss_func(predicted_labels, labels)
            loss += total_loss.item()
            print(f'Loss: {total_loss}')
            total_loss.backward()
            opt.step()
        train_loss.append(loss)
        if eval:
            train_best_threshold, _, train_pr_thr = evaluate_model(model=model, data_loader=train_loader, epochs=epoch, device=config.DEVICE, data_loader_name='train', loss_func=loss_func)
            test_best_threshold, test_loss, test_pr_thr = evaluate_model(model=model, data_loader=test_loader, epochs=epoch, device=config.DEVICE, data_loader_name='test', loss_func=loss_func)
            train_roc_thrs.append(train_best_threshold)
            test_roc_thrs.append(test_best_threshold)
            test_losses.append(test_loss)
            train_pr_thrs.append(train_pr_thr)
            test_pr_thrs.append(test_pr_thr)

        if epoch == 0:
            print(f'ETA {(time.time() - start) * epochs / 60} minutes')


    
    return model, train_loss, train_roc_thrs, test_roc_thrs, test_losses, train_pr_thrs, test_pr_thrs


def evaluate_model(model, data_loader, device, epochs, data_loader_name, loss_func):
    print('Evaluating model...')
    with torch.no_grad():

        model.eval()
        confusion_matrix = BinaryConfusionMatrix(threshold=0.5)
        precision_recall_curve_obj = BinaryPrecisionRecallCurve()
        true_labels = []
        predicted_probabilities = []
        total_loss = 0

        for images, labels in data_loader:
            images, labels = images.to(device, dtype=torch.float), labels.type(torch.LongTensor).to(device)
            # forward pass
            preds: Tensor = model(images)
            preds_prob = torch.nn.functional.softmax(preds, dim=1)
            ball_probs = preds_prob[:, 1] 
            loss = loss_func(preds_prob, labels)
            total_loss += loss.item()

            confusion_matrix.update(
                input=ball_probs.cpu(),
                target=labels.cpu(),
            )
            precision_recall_curve_obj.update(
                input=ball_probs.cpu(),
                target=labels.cpu(),
            )
            true_labels.extend(labels.cpu().numpy())
            predicted_probabilities.extend(ball_probs.cpu().numpy())

        print(confusion_matrix.compute())
        # Compute the Precision-Recall curve
        precision, recall, thrs_pr = precision_recall_curve_obj.compute()
        beta = 0.5
        fscore = ( 1 + beta**2 ) * ( precision * recall ) / ( (beta**2 * precision) + recall )
        best_pr_thres = thrs_pr[fscore.argmax()]
        print(f'Best Fbeta threshold {best_pr_thres}')
        with open(f'pr_curve_{data_loader_name}_diff.json', 'a') as f:
            
            json.dump({"time": time.time(), "epoch": epochs, "precision": precision.tolist(), "recall": recall.tolist(), "thrs": thrs_pr.tolist()}, f)
            f.write('\n')
        
        plt.style.use('dark_background')
        # Plot the Precision-Recall curve
        plt.figure()
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curve')
        plt.savefig(f'../figures/precision_recall_curve_{data_loader_name}_{epochs}.png')
        plt.close()


        # Compute the ROC curve
        fpr, tpr, thrs_roc = roc_curve(true_labels, predicted_probabilities)
          # Find the index of the threshold that maximizes TPR while minimizing FPR
        best_threshold_index = (tpr - fpr).argmax()

        # Get the best threshold
        best_threshold = thrs_roc[best_threshold_index]
        print("Best Threshold:", best_threshold)
        with open(f'roc_curve_{data_loader_name}_diff.json', 'a') as f:
            json.dump({"time": time.time(), "epoch": epochs, "fpr": fpr.tolist(), "tpr": tpr.tolist(), "thrs": thrs_roc.tolist()}, f)
            f.write('\n')

        roc_auc = roc_auc_score(true_labels, predicted_probabilities)
        print("ROC AUC:", roc_auc)

        # Plot the ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='b', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.savefig(f'../figures/roc_curve_{data_loader_name}_{epochs}.png')
        plt.close()

        return best_threshold, total_loss, best_pr_thres


def save_model(model, path):
    print(f"Saving model at {path}")
    torch.save(model, path)


def train(epochs, eval):
    image_list = import_data(
        ground_truth_path='../datasets/ground_truth_60x60_frame_diff/',
        ground_false_path='../datasets/ground_false_60x60_frame_diff/'
    )

    train_data, test_data = train_test_split(image_list, test_size=0.2, shuffle=True)
    print('Creating Dataset')

    transformations = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(random.randint(0, 360)),
        ]
    )
    train_dataset = ds.ImageDataset(
        train_data,
        transforms=transformations
    )
    train_dataset.get_channel_mean_std()

    test_dataset = ds.ImageDataset(
        test_data,
        transforms=transformations
    )
    test_dataset.means = train_dataset.means
    test_dataset.stds = train_dataset.stds


    # for i, e in enumerate(train_dataset.images):
    #     train_dataset.images[i].save(id=i)
    # for i, e in enumerate(test_dataset.images):
    #     test_dataset.images[i].save(id=i)
    
    train_dataset.get_channel_mean_std()

    BATCHES = 512
    print('Creating dataloader')

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCHES,
        shuffle=True,
        num_workers=1,
        pin_memory=config.PIN_MEMORY
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCHES,
        shuffle=True,
        num_workers=1,
        pin_memory=config.PIN_MEMORY
    )
    loss_func = torch.nn.CrossEntropyLoss()
    # Training
    EPOCHS = int(epochs)
    model, train_loss, train_roc_thrs, test_roc_thrs, test_loss, train_pr_thrs, test_pr_thrs = train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        loss_func=loss_func,
        learning_rate=.001,
        EPOCHS=EPOCHS,
        eval=eval,
    )



    # # Evaluation
    # train_roc_thrs, _ = evaluate_model(model, data_loader=train_loader, device=config.DEVICE, epochs=EPOCHS, data_loader_name='train', loss_func=loss_func)
    # test_roc_thrs, test_loss = evaluate_model(model, data_loader=test_loader, device=config.DEVICE, epochs=EPOCHS, data_loader_name='test', loss_func=loss_func)

    # Model saving
    print('saving model')
    save_model(model, './model.pth')
    print('model saved')
    return BATCHES, train_loss, train_roc_thrs, test_roc_thrs, test_loss,  train_pr_thrs, test_pr_thrs


if __name__ == '__main__':
    start = time.time()
    train_losses = [] 
    test_losses = []
    eval_bool = False
    for i in range(1):
        print(f'Using device: {config.DEVICE}')
        epochs = 23
        batches, train_loss, train_roc_thrs, test_roc_thrs, test_loss, train_pr_thrs, test_pr_thrs = train(epochs=epochs, eval=eval_bool)
        train_losses.append(train_loss)
        test_losses.append(test_loss)



        # Plot optimal ROC thresholds
        if eval_bool:
            plt.figure()
            plt.style.use('dark_background')
            plt.ylabel('Optimal ROC thresholds')
            plt.xlabel('Epochs')
            plt.plot(range(epochs), train_roc_thrs, '-o', label='train optimal thresholds')
            plt.plot(range(epochs), test_roc_thrs, '-o', label='test optimal thresholds')
            plt.legend()
            plt.savefig('../figures/optmimal_roc_thrs.png')
            plt.close()

            # Plot optimal PR thresholds
            plt.figure()
            plt.style.use('dark_background')
            plt.ylabel('Optimal PR-Curve thresholds')
            plt.xlabel('Epochs')
            plt.plot(range(epochs), train_pr_thrs, '-o', label='train optimal thresholds')
            plt.plot(range(epochs), test_pr_thrs, '-o', label='test optimal thresholds')
            plt.legend()
            plt.savefig('../figures/optimal_pr_thrs.png')
            plt.close()

            # Plot accuracy for thresholds
            plt.figure()
            plt.style.use('dark_background')
            plt.ylabel('Optimal PR-Curve thresholds')
            plt.xlabel('Epochs')
            plt.plot(range(epochs), train_pr_thrs, '-o', label='train optimal thresholds')
            plt.plot(range(epochs), test_pr_thrs, '-o', label='test optimal thresholds')
            plt.legend()
            plt.savefig('../figures/optimal_pr_thrs.png')
            plt.close()
            total = time.time() - start
            print(f'Time elapsed {total/60} minutes')
    if eval_bool:
        train_losses = np.array(train_losses)
        test_losses = np.array(test_losses)    
        mean_train_losses = np.mean(train_losses, axis=0)
        mean_test_losses = np.mean(test_losses, axis=0)

        plt.figure()
        plt.style.use('dark_background')
        plt.xlabel('EPOCHS')
        plt.ylabel('Training loss')
        plt.errorbar(range(epochs), mean_train_losses, fmt='-o', yerr=np.std(train_losses, axis=0))
        plt.legend()
        plt.savefig('../figures/training_loss_history_mean.png')
        plt.close()

        # Plot test loss
        plt.figure()
        plt.style.use('dark_background')
        plt.ylabel('Test loss')
        plt.xlabel('Epochs')
        plt.errorbar(range(epochs), mean_test_losses, fmt='-o',  yerr=np.std(test_losses, axis=0))
        plt.savefig('../figures/test_loss_mean.png')
        plt.close()
        print(f'Time elapsed {total/60} minutes')

