from data_provider.data_loader import Dataset_mekong, Dataset_mekong_cross, Dataset_mekong_phase_a
from torch.utils.data import DataLoader

data_dict = {
    'MeKong': Dataset_mekong_phase_a,
    'MeKong_Cross': Dataset_mekong_cross
}


def data_provider(args, flag, verbose=True):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.data == 'MeKong':
        data_set = Data(
            args=args,
            root_path=args.root_path,
            data_path=args.data1_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq
        )
        if verbose:
            print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.data == 'MeKong_Cross':
        data_set = Data(
            args=args,
            root_path=args.root_path,
            data1_path=args.data1_path,
            data2_path=args.data2_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq
        )
        if verbose:
            print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    else:
        raise NotImplementedError
