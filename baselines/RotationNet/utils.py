import torch
def collate_self(batch):
    batch_mod = {'sketch_img': [], 'sketch_boxes': [],
                 'sketch_label': [], 'sketch_path': []
                 }
    for i_batch in batch:
        batch_mod['sketch_img'].append(i_batch['sketch_img'])
        batch_mod['sketch_boxes'].append(torch.tensor(i_batch['sketch_boxes']).float())
        batch_mod['sketch_label'].append(i_batch['sketch_label'])
        batch_mod['sketch_path'].append(i_batch['sketch_path'])

    batch_mod['sketch_img'] = torch.stack(batch_mod['sketch_img'], dim=0)
    batch_mod['sketch_label'] = torch.tensor(batch_mod['sketch_label'])

    return batch_mod