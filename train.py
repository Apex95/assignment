import logging
import sys
import dataset
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from model import FurnitureTrf
from transformers import DataCollatorWithPadding

logger = logging.getLogger('furniture-trf-logger')
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger.info(f"Running on {device}")

PRECISION_RECALL_OFFSET = 0.1
MAX_LEN = 512
MAX_EPOCHS = 40
BATCH_SIZE = 128

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
backbone = DistilBertModel.from_pretrained('distilbert-base-uncased')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# furniture items detector
model = FurnitureTrf(backbone=backbone)

# freezing bert's layers -> these are pretty good
for param in model.trf.parameters():
    param.requires_grad = False

model.to(device)

# load & split datasets
train_dataset, val_dataset, neg_pos_ratio = dataset.load_dataset(
    json_path='data/formatted',
    tokenizer=tokenizer,
    max_len=MAX_LEN,
    split_percent=0.8
)


# construct training / validation loaders with dynamic batching
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=data_collator
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=data_collator
)


criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor(
    [PRECISION_RECALL_OFFSET * neg_pos_ratio]).to(device))
optimizer = torch.optim.AdamW(params=model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1, end_factor=0.01, total_iters=MAX_EPOCHS)

# for early stopping
min_validation_loss = None

for epoch in range(MAX_EPOCHS):
    training_loss = 0
    model.train()

    for batch_id, data in enumerate(train_loader):

        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)

        outputs = model(ids, mask)

        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

    logger.info(
        f"Training @{epoch} : {training_loss * BATCH_SIZE / len(train_loader)}")

    validation_loss = 0
    tps, tns, fps, fns = 0, 0, 0, 0
    model.eval()

    with torch.no_grad():
        for batch_id, data in enumerate(val_loader):
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            outputs = model(ids, mask)
            loss = criterion(outputs, targets)

            outputs = torch.sigmoid(outputs)

            for i in range(outputs.shape[0]):
                if outputs[i] < 0.5 and targets[i] < 0.5:
                    tns += 1

                if outputs[i] < 0.5 and targets[i] > 0.5:
                    fns += 1

                if outputs[i] > 0.5 and targets[i] < 0.5:
                    fps += 1

                if outputs[i] > 0.5 and targets[i] > 0.5:
                    tps += 1

            validation_loss += loss.item()

        logger.info(
            f"Validation @{epoch} : {validation_loss * BATCH_SIZE / len(val_loader)}")

        if min_validation_loss is None or validation_loss < min_validation_loss:
            min_validation_loss = validation_loss
            logger.info(
                f"Saving new model > checkpoints/{min_validation_loss}.dat")
            torch.save(model.state_dict(),
                       f"checkpoints/{min_validation_loss}.dat")

        logger.info(f'TP:{tps} TN:{tns} FP:{fps} FN:{fns}')

        scheduler.step()
