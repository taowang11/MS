import sys
from model import *
# from utils import *
from evalution import *
import torch.nn.functional as F
from nt_xent import NT_Xent
from model import set_seed
from sklearn.model_selection import KFold, GridSearchCV





# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    lossz = 0
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        batch1 = data.batch1.detach()
        edge = data.edge_index1.detach()
        xd = data.x1.detach()
        n = data.y.shape[0]  # batch
        optimizer.zero_grad()
        output, x_g, x_g1, output1 = model(data, data.x, data.edge_index, data.batch, xd, edge, batch1)
        loss_1 = criterion(output, data.y.view(-1, 1).float())
        criterion1 = NT_Xent(output.shape[0], 0.1, 1)
        loss_2 = criterion(output1, data.y.view(-1, 1).float())

        cl_loss = criterion1(x_g, x_g1)
        loss = loss_1 + 0.3 * cl_loss + loss_2
        # loss = loss_1
        loss.backward()
        optimizer.step()
        lossz = loss + lossz

    print('Train epoch: {} \tLoss: {:.6f}'.format(epoch, lossz))


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            batch1 = data.batch1.detach()
            edge = data.edge_index1.detach()
            xd = data.x1.detach()
            output, x_g, x_g1, output1 = model(data, data.x, data.edge_index, data.batch, xd, edge, batch1)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.cpu()), 0)
    return total_labels, total_preds


if __name__ == "__main__":
    set_seed(42)
    cuda_name = "cuda:0"
    print('cuda_name:', cuda_name)

    NUM_EPOCHS = 300
    LR = 0.0005

    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)
    root = 'data'  # data_External/data_rlm/data/noH
    processed_train = root + '/train.pth'
    processed_test = root + '/test.pth'
    data_listtrain = torch.load(processed_train)
    data_listtest = torch.load(processed_test)

    


    def custom_batching(data_list, batch_size):
        for i in range(0, len(data_list), batch_size):
            yield data_list[i:i + batch_size]


    best_results = []

    batchestrain = list(custom_batching(data_listtrain, 256))
    batchestrain1 = list()
    for batch_idx, data in enumerate(batchestrain):
        data = collate_with_circle_index(data)
        data.edge_attr = None

        batchestrain1.append(data)
    batchestest = list(custom_batching(data_listtest, 1000))
    batchestest1 = list()
    for batch_idx, data in enumerate(batchestest):
        data = collate_with_circle_index(data)
        data.edge_attr = None
        batchestest1.append(data)

    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = MS_BACL().cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    max_auc = 0

    model_file_name = 'model' + '.pt'
    result_file_name = 'result' + '.csv'
    for epoch in range(NUM_EPOCHS):
        train(model, device, batchestrain1, optimizer, epoch + 1)
        G, P = predicting(model, device, batchestest1)

        auc, acc, precision, recall, f1_scroe, mcc = metric(G, P)
        ret = [auc, acc, precision, recall, f1_scroe, mcc]
        if auc > max_auc:
            max_auc = auc
            torch.save(model.state_dict(), model_file_name)
            with open(result_file_name, 'w') as f:
                f.write(','.join(map(str, ret)))
            ret1 = [auc, acc, precision, recall, f1_scroe, mcc]

        print(
            'test---------------------------  auc:%.4f\t acc:%.4f\t precision:%.4f\t recall:%.4f\tf1_scroe:%.4f\t mcc:%.4f' % (
                auc, acc, precision, recall, f1_scroe, mcc))

    print('Maximum acc found. Model saved.')
    best_results.append({
        'auc, acc, precision, recall, f1_scroe, mcc': (
            ret1)

    })
for i, result in enumerate(best_results, 1):
    print(f"Fold {i}:")
    print(f"Best auc, acc, precision, recall, f1_scroe, mcc: {result['auc, acc, precision, recall, f1_scroe, mcc']}")
