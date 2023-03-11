import argparse
import json
import os

import numpy as np
from sklearn.metrics import auc, f1_score, roc_curve
import torch
from torch import optim
from torch.nn import functional as F
from torch.optim import lr_scheduler

from models.encoders import MethEncoder, SPECTEncoder
from models.Integrator import MethSpectIntegrator
from utils.dataloader import PPMIDataset

MODELS = "data/models/ppmi/"
RESULTS = "data/results/ppmi"
VISUALISATIONS = "data/results/ppmi/visualisations"

if not os.path.exists(MODELS):
    os.mkdir(MODELS)

if not os.path.exists(RESULTS):
    os.mkdir(RESULTS)

if not os.path.exists(VISUALISATIONS):
    os.mkdir(VISUALISATIONS)


def parse_args():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--learning_rate', type=float, default=3e-05)
    parser.add_argument('--experiment', type=str, help="name of experiment")
    parser.add_argument('--missing_samples', type=float)
    parser.add_argument('--missing_data', type=float)
    parser.add_argument('--validation_split', type=float, default=0.2)
    parser.add_argument('--feature_size', type=int, help="number of features")
    parser.add_argument('--dropout_keep_prob', type=float, default=1.)
    parser.add_argument('--block_shape', type=int, default=64)
    parser.add_argument('--blocks', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--embedding_size', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save', action='store_true', default=False,
                        help='save result and model')
    parser.add_argument('--num_datasets', type=int, default=1)
    parser.add_argument('--meth', action='store_true', default=False)
    parser.add_argument('--spect', action='store_true', default=False)
    parser.add_argument('--ignore_attention', action='store_true', default=False)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--classification', action='store_true', default=False)
    parser.add_argument('--early_stop_thresh', type=float, default=0)
    parser.add_argument('--early_stop_epochs', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--balance', action='store_true', default=False)
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--aug_frac', type=float, default=0.2)
    parser.add_argument('--folds', type=int, default=1)
    parser.add_argument('--kfold', action='store_true', default=False)

    args = parser.parse_args()
    return args

def get_auc(label, pred):
    false_positive_rate, true_positive_rate, _ = roc_curve(label, pred)
    return auc(false_positive_rate, true_positive_rate)


def get_batch(device, meth, spect, use_meth, use_spect, num_datasets):
    if num_datasets == 1:
        if use_meth:
            meth = meth.float().to(device)
            data = (meth)
        elif use_spect:
            spect = spect.to(device)
            data = (spect)
    elif num_datasets == 2:
        if use_meth and use_spect:
            meth = meth.float().to(device)
            spect = spect.to(device)
            data = (meth, spect)
    else:
        raise NotImplementedError

    return data


def train(args, model, device, train_loader, optimizer, scheduler, epoch, use_meth, use_spect, num_datasets, criterion):
    model.train()
    train_loss = 0
    train_correct = 0

    for batch_idx, (meth, spect, target) in enumerate(train_loader):

        data = get_batch(device, meth, spect, use_meth, use_spect, num_datasets)
        if args.classification:
            target = target.long().to(device)
        else:
            target = target.float().to(device)

        correct = 0
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        if args.classification:
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            train_correct += (correct / len(target))

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Correct: {:.2f}'.format(
                    epoch, batch_idx * len(meth), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), 100. * correct / len(target)))
        else:
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(meth), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    avg_loss = train_loss / len(train_loader)
    avg_acc = train_correct / len(train_loader)
    print('Train - average loss: {:.6f}\t average accuracy: {:.6f}'.format(avg_loss, avg_acc))

    return avg_loss, avg_acc


def test(args, model, device, test_loader, use_meth, use_spect, num_datasets, criterion):
    model.eval()
    test_loss = 0
    correct = 0

    all_pred = []
    all_label = []

    with torch.no_grad():
        for (meth, spect, target) in test_loader:

            data = get_batch(device, meth, spect, use_meth, use_spect, num_datasets)
            if args.classification:
                target = target.long().to(device)
            else:
                target = target.float().to(device)

            output = model(data)
            test_loss += criterion(output, target).item()

            if args.classification:
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

                all_pred += list(pred.cpu().numpy())
                all_label += list(target.cpu().numpy())

    test_loss /= len(test_loader)
    test_acc = correct / len(test_loader.dataset)
    test_auc = get_auc(all_label, all_pred)

    if args.classification:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), F1: {:.4f}, AUC: {:.4f}\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * test_acc, f1_score(all_label, all_pred), test_auc))
    else:
        print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

    return test_loss, test_acc, test_auc


def evaluate(args, model, device, eval_loader, use_meth, use_spect, num_datasets, criterion):
    model.eval()
    eval_loss = 0
    correct = 0
    labels = []
    predictions = []

    all_pred = []
    all_label = []

    with torch.no_grad():
        for (meth, spect, target) in eval_loader:

            data = get_batch(device, meth, spect, use_meth, use_spect, num_datasets)
            if args.classification:
                target = target.long().to(device)
            else:
                target = target.float().to(device)

            output = model(data)
            eval_loss += criterion(output, target).item()

            labels += list(target.cpu().numpy().flatten())
            predictions += list(output.cpu().numpy().flatten())
            if args.classification:
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

                all_pred += list(pred.cpu().numpy())
                all_label += list(target.cpu().numpy())

    eval_loss /= len(eval_loader)
    acc = correct / len(eval_loader.dataset)
    eval_auc = get_auc(all_label, all_pred)

    if args.classification:
        print('\nEvaluation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), F1: {:.4f}, AUC: {:.4f}\n'.format(
            eval_loss, correct, len(eval_loader.dataset), 100. * acc, f1_score(all_label, all_pred), eval_auc))
    else:
        print('\nEvaluation set: Average loss: {:.4f}\n'.format(eval_loss))
        acc = eval_loss

    labels = list(map(float, np.array(labels).flatten()))
    predictions = list(map(float, np.array(predictions).flatten()))

    return labels, predictions, eval_loss, acc, eval_auc

def visualise_weights(args, model, device, vis_loader, use_meth, use_spect, num_datasets, num_examples=100):
    model.eval()
    with torch.no_grad():
        for img_idx, (meth, spect, target) in enumerate(vis_loader):

            if img_idx == num_examples:
                break

            data = get_batch(device, meth, spect, use_meth, use_spect, num_datasets)
            target = target.float().to(device)

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            if num_datasets == 1:
                to_save = {
                    "weights": model.get_weights(),
                    "modality_1": data.cpu().numpy(),
                    "label": target.cpu().numpy().squeeze(),
                    "prediction": pred.cpu().numpy().squeeze()
                }
                np.save(os.path.join(VISUALISATIONS, "{}_example-{}.npy".format(args.experiment, img_idx)), to_save)

            if num_datasets == 2:
                to_save = {
                    "weights": model.get_weights(),
                    "modality_1": data[0].cpu().numpy(),
                    "modality_2": data[1].cpu().numpy(),
                    "label": target.cpu().numpy().squeeze(),
                    "prediction": pred.cpu().numpy().squeeze()
                }
                np.save(os.path.join(VISUALISATIONS, "{}_example-{}.npy".format(args.experiment, img_idx)), to_save)


def main():

    ############################################################################
    ## FIXME                                                                  ##
    ## 1) regression is broken - need to fix augmentation to account for this ##
    ############################################################################

    args = parse_args()

    if not args.kfold and args.folds != 1:
        raise RuntimeError("kfolds set to false but number of folds greater than 1")

    if args.folds != 1 and args.folds != 5:
        raise RuntimeError("only have support for 5 fold cross validation -- {}".format(args.folds))

    if args.classification:

        if not args.kfold:
            train_dataset = PPMIDataset(train=True,
                                        classification=args.classification,
                                        balance=args.balance,
                                        augmentation=args.augment,
                                        fraction=args.aug_frac)
            test_dataset = PPMIDataset(train=False,
                                    classification=args.classification,
                                    balance=False)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=True)
            visualisation_loader = torch.utils.data.DataLoader(test_dataset,
                                                            batch_size=1,
                                                            shuffle=True)
    else:
        raise NotImplementedError("Regression not supported.")

    device = torch.device("cuda" if args.cuda else "cpu")

    all_summary_results = []
    all_acc_results = []
    best_acc = []
    best_auc = []

    if args.classification:
        criterion = F.nll_loss
    else:
        raise NotImplementedError("Regression not supported.")

    for run in range(args.runs):

        vis_model_path = os.path.join(MODELS, "model_visualisations_{}_ms-{}_md-{}_bs-{}_ep-{}_feats-{}_heads-{}_emb-{}_run-{}.pt".format(args.experiment,
                                                                                                                                          args.missing_data, args.missing_samples, args.batch_size,
                                                                                                                                          args.epochs, args.feature_size, args.num_heads,
                                                                                                                                          args.embedding_size, run))

        summary_results = {
            "train_loss": [],
            "test_loss": [],
            "train_acc": [],
            "test_acc": [],
            "test_auc": [],
            "args": vars(args)
        }

        best_acc_folds = []
        best_auc_folds = []

        for fold in range(args.folds):

            print("run: {} fold: {}".format(run, fold))

            if args.kfold:
                train_dataset = PPMIDataset(train=True,
                                            classification=args.classification,
                                            balance=args.balance,
                                            augmentation=args.augment,
                                            fraction=args.aug_frac,
                                            folds=args.folds,
                                            fold=fold)
                test_dataset = PPMIDataset(train=False,
                                           classification=args.classification,
                                           balance=False,
                                           folds=args.folds,
                                           fold=fold)
                train_loader = torch.utils.data.DataLoader(train_dataset,
                                                           batch_size=args.batch_size,
                                                           shuffle=True)
                test_loader = torch.utils.data.DataLoader(test_dataset,
                                                          batch_size=args.batch_size,
                                                          shuffle=True)
                visualisation_loader = torch.utils.data.DataLoader(test_dataset,
                                                                   batch_size=1,
                                                                   shuffle=True)

            if args.num_datasets == 2:
                model = MethSpectIntegrator(MethEncoder, SPECTEncoder, **vars(args))
            else:
                raise NotImplementedError

            model.to(device)
            print(model)

            print("Number of parameters:", sum([x.numel() for x in model.parameters()]))

            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-3)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader), eta_min=1e-07)

            valid_loss_min = np.inf
            epochs_since_decrease = 0

            for epoch in range(1, args.epochs + 1):

                if epochs_since_decrease == args.early_stop_epochs:
                    print("Accuracy has not increased by {:.2f} in {} epochs, breaking training".format(args.early_stop_thresh, args.early_stop_epochs))
                    break

                train_loss, train_acc = train(args, model, device, train_loader, optimizer, scheduler, epoch, args.meth, args.spect, args.num_datasets, criterion)
                test_loss, test_acc, test_auc = test(args, model, device, test_loader, args.meth, args.spect, args.num_datasets, criterion)

                summary_results['train_loss'].append(float(train_loss))
                summary_results['test_loss'].append(float(test_loss))
                summary_results['train_acc'].append(float(train_acc))
                summary_results['test_acc'].append(float(test_acc))
                summary_results['test_auc'].append(float(test_auc))


                if (test_loss < valid_loss_min) and args.save:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving visualisation model ...'.format(
                        valid_loss_min,
                        test_loss))
                    torch.save(model.state_dict(), vis_model_path)
                    valid_loss_min = test_loss
                    epochs_since_decrease = 0
                else:
                    epochs_since_decrease += 1


            print("Evaluating best model...")

            if args.num_datasets == 2:
                model = MethSpectIntegrator(MethEncoder, SPECTEncoder, **vars(args)).to(device)
            else:
                raise NotImplementedError
            model.load_state_dict(torch.load(vis_model_path))

            labels, predictions, eval_loss, eval_acc, eval_auc = evaluate(args, model, device, test_loader, args.meth, args.spect, args.num_datasets, criterion)

            best_acc_folds.append(eval_acc)
            best_auc_folds.append(eval_auc)

        print("Run {}\t avg acc: {} avg auc: {}".format(run, np.mean(best_acc_folds), np.mean(best_auc_folds)))

        best_acc.append(np.mean(best_acc_folds))
        best_auc.append(np.mean(best_auc_folds))

        labels = list(map(float, np.array(labels).flatten()))
        predictions = list(map(float, np.array(predictions).flatten()))

        acc_results = {
            "preds": predictions,
            "labels": labels,
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
            "eval_auc": eval_auc
        }

        all_summary_results.append(summary_results)
        all_acc_results.append(acc_results)


    if not args.ignore_attention:

        visualise_weights(args, model, device, visualisation_loader, args.meth, args.spect, args.num_datasets)

    if args.save:
        save_results(args, all_summary_results, all_acc_results)

    print("-- {} -- Mean evaluation accuracy: {:.4f} +/- {:.4f}\t AUC: {:.4f} +/- {:.4f}".format(args.experiment, 100.*np.mean(best_acc), 100.*np.std(best_acc), np.mean(best_auc), np.std(best_auc)))


def save_results(args, results, acc_results):
    with open(os.path.join(RESULTS, "losses_{}_ms-{}_md-{}_bs-{}_ep-{}_feats-{}_heads-{}_emb-{}.json".format(args.experiment,
                                                                                                             args.missing_data,
                                                                                                             args.missing_samples,
                                                                                                             args.batch_size,
                                                                                                             args.epochs,
                                                                                                             args.feature_size,
                                                                                                             args.num_heads,
                                                                                                             args.embedding_size)), "w+") as fd:
        json.dump(results, fd)
    with open(os.path.join(RESULTS, "control_predictions_{}_ms-{}_md-{}_bs-{}_ep-{}_feats-{}_heads-{}_emb-{}.json".format(args.experiment,
                                                                                                                          args.missing_data,
                                                                                                                          args.missing_samples,
                                                                                                                          args.batch_size,
                                                                                                                          args.epochs,
                                                                                                                          args.feature_size,
                                                                                                                          args.num_heads,
                                                                                                                          args.embedding_size)), "w+") as fd:
        json.dump(acc_results, fd)


if __name__ == "__main__":
    main()
