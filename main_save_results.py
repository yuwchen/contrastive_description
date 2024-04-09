from load import *
from tqdm import tqdm
import torchmetrics


seed_everything(hparams['seed'])

bs = hparams['batch_size']
dataloader = DataLoader(dataset, bs, shuffle=True, num_workers=16, pin_memory=True)

print("Loading model...")
device='cpu'
#device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using {} ...".format(device))
# load model
model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)
model.eval()
model.requires_grad_(False)
print('Images from {} ...'.format(GEO_DIR))
print("Encoding descriptions...")

description_encodings = compute_description_encodings(model)
label_encodings = compute_label_encodings(model)


des_label = {}
for i, (k, v) in enumerate(description_encodings.items()):
    des_label[k]=i

print('label and description:',des_label)

print("Evaluating...")
lang_accuracy_metric = torchmetrics.Accuracy().to(device)
lang_accuracy_metric_top5 = torchmetrics.Accuracy(top_k=5).to(device)


for batch_number, batch in enumerate(tqdm(dataloader)):
    

    images, raw_labels, filename = batch
    
    images = images.to(device)
    image_encodings = model.encode_image(images)
    image_encodings = F.normalize(image_encodings)
    
    image_labels_similarity = image_encodings @ label_encodings.T    
    
    image_description_similarity = [None]*n_classes
    image_description_similarity_cumulative = [None]*n_classes
    
    for i, (k, v) in enumerate(description_encodings.items()): 
        

        dot_product_matrix = image_encodings @ v.T
        image_description_similarity[i] = dot_product_matrix
        image_description_similarity_cumulative[i] = aggregate_similarity(image_description_similarity[i])
        
        
    # create tensor of similarity means
    cumulative_tensor = torch.stack(image_description_similarity_cumulative,dim=1)
    descr_predictions = cumulative_tensor.argmax(dim=1)
    
    labels = [des_label[label] for label in raw_labels]
    labels = torch.tensor(labels, dtype=torch.int32).to(device)
    
    if batch_number==0:
        labels_all = labels
        results = cumulative_tensor.softmax(dim=-1)
        filename_list = [filename]
    else:
        labels_all = torch.cat([labels_all, labels], dim=0)
        results = torch.cat([results, cumulative_tensor.softmax(dim=-1)], dim=0)
        filename_list.append(filename)

    lang_acc = lang_accuracy_metric(cumulative_tensor.softmax(dim=-1), labels)
    lang_acc_top5 = lang_accuracy_metric_top5(cumulative_tensor.softmax(dim=-1), labels)


print("\n")

lang_accuracy_metric_top2 = torchmetrics.Accuracy(top_k=2).to(device)
lang_accuracy_metric_top3 = torchmetrics.Accuracy(top_k=3).to(device)
lang_accuracy_metric_top4 = torchmetrics.Accuracy(top_k=4).to(device)
lang_acc = lang_accuracy_metric(results, labels_all)
lang_acc_top2 = lang_accuracy_metric_top2(results, labels_all)
lang_acc_top3 = lang_accuracy_metric_top3(results, labels_all)
lang_acc_top4 = lang_accuracy_metric_top4(results, labels_all)
lang_acc_top5 = lang_accuracy_metric_top5(results, labels_all)

print('top1:', lang_acc)
print('top2:', lang_acc_top2)
print('top3:', lang_acc_top3)
print('top4:', lang_acc_top4)
print('top5:', lang_acc_top5)

torch.save(filename_list, os.path.join(hparams['output_dir'], 'filename_list.pt'))
torch.save(labels_all, os.path.join(hparams['output_dir'],'labels_all.pt'))
torch.save(results, os.path.join(hparams['output_dir'],' predictions.pt'))
torch.save(label_to_classname,os.path.join(hparams['output_dir'],'label_to_classname.pt'))

