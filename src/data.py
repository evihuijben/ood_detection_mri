from monai import transforms
from monai.data import Dataset, DataLoader 
from pathlib import Path


import os

class CustomDataset(Dataset):
    def __init__(self, data, transform):
        super().__init__(data, transform)

    def __getitem__(self, index):
        out = self._transform(index)
        out['path'] = self.data[index]['image']
        out['ID'] = Path(self.data[index]['image']).name.split('.')[0].split('_')[0]
        out['slice'] = int(Path(self.data[index]['image']).name.split('.')[0].split('_')[1])
        return out
        
def get_data_dicts_mood(data_dir, args):
    
    data_list = sorted(os.listdir(data_dir))
    if args.which_slices != 'all':
        if args.which_slices == 'mid':
            n_included = 1
        else:
            n_included = int(args.which_slices.split('_')[-1])
        
        start = (args.total_slices - n_included) // 2
        end = start + n_included
        print('Selecting slices from', start, 'to', end)
        data_list = [x for x in data_list if int(x.split('.')[0].split('_')[1]) in range(start, end)]

    data_dicts = [{"image": (os.path.join(data_dir, name))} for name in data_list]
    print( len(data_dicts), 'images used from', data_dir)
    return data_dicts


def get_data_loader(data_dir, args, shuffle=True, drop_last=False):

    tf = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]) if args.is_grayscale else lambda x: x,
            transforms.ToTensord(keys=["image"]),
        ]
    )

    data_dicts = get_data_dicts_mood(data_dir, args)
    
    ds = CustomDataset(data=data_dicts, transform=tf)


    loader = DataLoader(ds,
                        batch_size=args.batch_size,
                        shuffle=shuffle,
                        num_workers=args.num_workers,
                        drop_last=drop_last,
                        pin_memory=False)

    return loader


