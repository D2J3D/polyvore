import torch
from torch.utils.data import DataLoader

from inputs import CustomDataset, batch_with_dynamic_pad

# Пример синтетических данных
sample_data = [
    {
        'set_id': 'set_1',
        'image_feature/0': 'image_data_1_0',
        'image_feature/1': 'image_data_1_1',
        'image_index': [0, 1],
        'caption_feature': [1, 2, 3, 4],
        'likes': 10
    },
    {
        'set_id': 'set_2',
        'image_feature/0': 'image_data_2_0',
        'image_feature/1': 'image_data_2_1',
        'image_index': [0, 1],
        'caption_feature': [1, 2, 3],
        'likes': 5
    }
]

# Настройки для тестирования
set_id = 'set_id'
image_feature = 'image_feature'
image_index = 'image_index'
caption_feature = 'caption_feature'
number_set_images = 2

# Создание объекта Dataset
dataset = CustomDataset(sample_data, set_id, image_feature, image_index, caption_feature, number_set_images)

# Создание DataLoader для пакетной загрузки данных
dataloader = batch_with_dynamic_pad(dataset, batch_size=2)

# Проверка вывода данных
for batch in dataloader:
    set_ids, encoded_images, image_ids, captions_padded, masks, likes = batch
    
    print("Set IDs:", set_ids)
    print("Encoded Images:", encoded_images)
    print("Image IDs:", image_ids)
    print("Captions (padded):", captions_padded)
    print("Masks:", masks)
    print("Likes:", likes)
