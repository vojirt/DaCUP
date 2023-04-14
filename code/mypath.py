class Path(object):
    @staticmethod
    def dataset_root_dir(dataset):
        if dataset in ["cityscapes", "cityscapes_2class", "cityscapes_all_p1"]:
            return '/mnt/datagrid/public_datasets/CityScapes/'     # folder that contains leftImg8bit/
        if dataset == 'LaF':
            return '/mnt/datagrid/public_datasets/lost_and_found/'     # folder that contains leftImg8bit/
        if dataset == "bdd100k_2class":
            return '/mnt/datasets/BerkeleyDeepDriveDataset/bdd100k/' 
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
