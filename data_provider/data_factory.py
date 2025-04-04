from data_provider.data_loader import Dataset_mekong, Dataset_mekong_cross, Dataset_mekong_phase_a
from torch.utils.data import DataLoader

data_dict = {
    'Initial': Dataset_mekong,
    'PhaseA': Dataset_mekong_phase_a,
}


def data_provider(args, flag, verbose=True):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    data_set = Data(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
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
