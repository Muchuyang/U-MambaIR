##### This paper is currently under submission. If it gets accepted, we will release the complete code as soon as possible. #####
from torch.utils.data import DataLoader



def getDataloader(dataset, opt, is_Train=True):
    if is_Train:
        test_load = DataLoader(dataset=dataset, batch_size=opt.batch_size,
                               shuffle=opt.shuffle, num_workers=opt.num_workers,
                               drop_last=opt.drop_last)
    else:
        test_load = DataLoader(dataset=dataset, batch_size=opt.batch_size,
                               shuffle=False, num_workers=opt.num_workers,
                               drop_last=False)
    return test_load
