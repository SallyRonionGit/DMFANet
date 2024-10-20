
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = "norm"
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.root_dir = './levir'
        elif data_name =='WHU':
            self.root_dir = './WHU'
        elif data_name =='DSIFN':
            self.root_dir = './DSIFN'
        elif data_name == 'quick_start':
            self.root_dir = './samples/'
        elif data_name =='LEVIR_SAMPLE':
            self.root_dir = './levir_sample'    
        elif data_name =='S2Looking':
            self.root_dir = './S2Looking'
        elif data_name =='SYSU':
            self.root_dir = './SYSU' 
        elif data_name =='EGY':
            self.root_dir = './EGY'
        elif data_name =='BCDD':
            self.root_dir = './BCDD'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

