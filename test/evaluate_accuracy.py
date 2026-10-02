import os, random
import numpy as np
import torch
import yacs.config
from collections import defaultdict

from module.seismic_data import SeismicVolume, Coordinate
from network.coordi_network import get_gen_model
from utils.project import get_project_root
from test.create_validation_figures import extract_random_line, predict_section

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

random.seed(42)
np.random.seed(42)

target_dim = 256
crop_size = (768, 512)
device = torch.device('cuda:9')
n_samples = 100

config_file = os.path.join(get_project_root(), 'config', 'config_lithofacies.yaml')
with open(config_file, 'rt') as f:
    CF = yacs.config.load_cfg(f)

data_pool = CF.DATASET.DATA_POOL
volume_tag = CF.DATASET.VOLUME_TAG
facies_tag = CF.DATASET.FACIES_TAG
valid_idx = CF.DATASET.VALID_IDX
vdt = CF.DATASET.VDT
tdt = 25.0

print('Loading data...')
volume = np.load(os.path.join(get_project_root(), 'data', data_pool, volume_tag + '.npy'), mmap_mode='r')[:, valid_idx[0]:valid_idx[1]]
facies = np.load(os.path.join(get_project_root(), 'data', data_pool, facies_tag + '.npy'), mmap_mode='r')[:, valid_idx[0]:valid_idx[1]]
inst_phase = np.load(os.path.join(get_project_root(), 'data', data_pool, volume_tag + '_inst_phase.npy'), mmap_mode='r')[:, valid_idx[0]:valid_idx[1]]
inst_freq = np.load(os.path.join(get_project_root(), 'data', data_pool, volume_tag + '_inst_freq.npy'), mmap_mode='r')[:, valid_idx[0]:valid_idx[1]]

nz, nx, ny = volume.shape
print('Volume shape:', volume.shape)

volume_sv = SeismicVolume()
volume_sv.clean_data(shape=(nz, nx, ny))
volume_sv.set_coordi(Coordinate(0, 0), Coordinate(0, ny - 1), Coordinate(nx - 1, 0), Coordinate(nx - 1, ny - 1))
volume_sv.data = volume

facies_sv = SeismicVolume()
facies_sv.clean_data(shape=(nz, nx, ny))
facies_sv.set_coordi(Coordinate(0, 0), Coordinate(0, ny - 1), Coordinate(nx - 1, 0), Coordinate(nx - 1, ny - 1))
facies_sv.data = facies

inst_phase_sv = SeismicVolume()
inst_phase_sv.clean_data(shape=(nz, nx, ny))
inst_phase_sv.set_coordi(Coordinate(0, 0), Coordinate(0, ny - 1), Coordinate(nx - 1, 0), Coordinate(nx - 1, ny - 1))
inst_phase_sv.data = inst_phase

inst_freq_sv = SeismicVolume()
inst_freq_sv.clean_data(shape=(nz, nx, ny))
inst_freq_sv.set_coordi(Coordinate(0, 0), Coordinate(0, ny - 1), Coordinate(nx - 1, 0), Coordinate(nx - 1, ny - 1))
inst_freq_sv.data = inst_freq

network = get_gen_model(CF, additional_channel=0).to(device)
checkpoint_path = os.path.join(get_project_root(), 'checkpoint', 'lithofacies_prediction_25.0_pat_044')
state = torch.load(checkpoint_path, map_location='cpu')
network.load_state_dict(state['network'])
network.eval()
print('Loaded checkpoint')

CLASS_NAMES = {0: 'Unclassified', 2: 'Basement', 3: 'Igneous', 4: 'Shale', 5: 'Sand'}

nt = crop_size[1]
all_gt = []
all_pred = []

print('')
print('Evaluating', n_samples, 'random sections...')
with torch.no_grad():
    for idx in range(n_samples):
        if (idx + 1) % 20 == 0:
            print('  Processing', idx + 1, '/', n_samples, '...')
        
        while True:
            vt, scoordi, ecoordi = extract_random_line(volume_sv, nt, vdt, tdt, mode='bilinear')
            ft, _, _ = extract_random_line(facies_sv, nt, vdt, tdt, scoordi=scoordi, ecoordi=ecoordi, mode='nearest')
            ft = np.round(ft).astype(np.int32)
            sz = random.randint(0, nz - crop_size[0])
            ez = sz + crop_size[0]
            if np.any(ft[sz:ez] != 0):
                break
        
        vt = vt[sz:ez]
        ft = ft[sz:ez]
        
        ip, _, _ = extract_random_line(inst_phase_sv, nt, vdt, tdt, scoordi=scoordi, ecoordi=ecoordi, mode='bilinear')
        ip = ip[sz:ez]
        if_, _, _ = extract_random_line(inst_freq_sv, nt, vdt, tdt, scoordi=scoordi, ecoordi=ecoordi, mode='bilinear')
        if_ = if_[sz:ez]
        
        rms = (vt * vt).mean() ** 0.5
        if rms > 0:
            vt = vt / rms * 0.15
        
        vt_tensor = torch.tensor(vt[None, None], dtype=torch.float32, device=device)
        info_tensor = torch.tensor(np.stack([ip, if_])[None], dtype=torch.float32, device=device)
        
        fo = predict_section(network, vt_tensor, info_tensor, device, target_dim)
        fo[ft == 0] = 0
        
        mask = ft != 0
        all_gt.extend(ft[mask].flatten())
        all_pred.extend(fo[mask].flatten())

all_gt = np.array(all_gt)
all_pred = np.array(all_pred)

print('')
print('Total evaluated pixels:', len(all_gt))

overall_acc = accuracy_score(all_gt, all_pred)
print('')
print('=' * 60)
print('OVERALL ACCURACY: {:.2f}%'.format(overall_acc * 100))
print('=' * 60)

print('')
print('=' * 60)
print('PER-CLASS METRICS')
print('=' * 60)

classes_present = sorted(set(all_gt) | set(all_pred))
classes_present = [c for c in classes_present if c != 0]

for cls in classes_present:
    gt_cls = all_gt == cls
    pred_cls = all_pred == cls
    
    tp = np.sum(gt_cls & pred_cls)
    fp = np.sum((~gt_cls) & pred_cls)
    fn = np.sum(gt_cls & (~pred_cls))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    n_gt = np.sum(gt_cls)
    n_pred = np.sum(pred_cls)
    
    print('')
    print(CLASS_NAMES.get(cls, 'Class ' + str(cls)) + ':')
    print('  Ground Truth count: {:,} ({:.1f}%)'.format(n_gt, n_gt/len(all_gt)*100))
    print('  Prediction count:   {:,} ({:.1f}%)'.format(n_pred, n_pred/len(all_pred)*100))
    print('  Precision: {:.1f}%'.format(precision * 100))
    print('  Recall:    {:.1f}%'.format(recall * 100))
    print('  F1-Score:  {:.1f}%'.format(f1 * 100))

print('')
print('=' * 60)
print('CONFUSION MATRIX')
print('=' * 60)

cm = confusion_matrix(all_gt, all_pred, labels=classes_present)
print('')
print('Rows: Ground Truth, Columns: Prediction')
print('Classes:', [CLASS_NAMES.get(c, c) for c in classes_present])
print('')

header = '          ' + ''.join(['{:>10}'.format(CLASS_NAMES.get(c, c)[:8]) for c in classes_present])
print(header)

for i, cls in enumerate(classes_present):
    row = '{:<10}'.format(CLASS_NAMES.get(cls, cls)[:8]) + ''.join(['{:>10,}'.format(cm[i,j]) for j in range(len(classes_present))])
    print(row)

print('')
print('=' * 60)
print('NORMALIZED CONFUSION MATRIX (% of Ground Truth)')
print('=' * 60)

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

print(header)
for i, cls in enumerate(classes_present):
    row = '{:<10}'.format(CLASS_NAMES.get(cls, cls)[:8]) + ''.join(['{:>10.1f}'.format(cm_norm[i,j]) for j in range(len(classes_present))])
    print(row)

f1_weighted = f1_score(all_gt, all_pred, labels=classes_present, average='weighted')
f1_macro = f1_score(all_gt, all_pred, labels=classes_present, average='macro')

print('')
print('=' * 60)
print('SUMMARY')
print('=' * 60)
print('Overall Accuracy:  {:.2f}%'.format(overall_acc * 100))
print('Weighted F1-Score: {:.2f}%'.format(f1_weighted * 100))
print('Macro F1-Score:    {:.2f}%'.format(f1_macro * 100))
