import paddle

class NewsData(paddle.io.Dataset):
    def __init__(self, data_path, mode = 'train'):
        is_test = True if mode == 'test' else False
        self.label_map = { item:index for index, item in enumerate(self.label_list) }
        self.examples = self.read_file(data_path, is_test)

    def read_file(self, data_path, is_test):
        examples = []
        with open(data_path, 'r', encoding = 'utf-8') as f:
            for line in f:
                if is_test:
                    text = line.strip()
                    examples.append((text, ))
                else:
                    text, label = line.strip('\n').split('\t')
                    label = self.label_map[label]
                    examples.append((text, label))
        return examples
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
    def __len__(self):
        return len(self.examples)
    
    @property
    def label_list(self):
        return ['财经', '彩票', '房产', '股票', '家居', '教育', '科技', '社会', '时尚', '时政', '体育', '星座', '游戏', '娱乐']